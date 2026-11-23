import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# Prioritize CUDA GPU device allocation for maximum acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

if torch.cuda.is_available():
    print(f"GPU acceleration is ACTIVE! Running on: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Running on CPU with JIT-accelerated operations.")

def generate_mesh_2d(Nx, Ny):
    x = np.linspace(0, 1, Nx + 1, dtype=np.float32)
    y = np.linspace(0, 1, Ny + 1, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    j, i = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
    n0 = (j * (Nx + 1) + i).ravel()
    n1 = n0 + 1
    n2 = n0 + (Nx + 1)
    n3 = n2 + 1

    elements = np.empty((Nx * Ny * 2, 3), dtype=np.int64)
    elements[0::2] = np.stack([n0, n1, n2], axis=1)
    elements[1::2] = np.stack([n1, n3, n2], axis=1)

    return nodes, elements

nodes, elements = generate_mesh_2d(4, 4)
print(f"Generated structured grid with {len(nodes)} nodes and {len(elements)} triangles.")

def build_direct_csr(rows, cols, vals, num_nodes, device):
    indices = torch.stack([rows, cols])
    coo = torch.sparse_coo_tensor(indices, vals, size=(num_nodes, num_nodes), device=device).coalesce()
    return coo.to_sparse_csr()

def compute_mass_matrix_2d(nodes, elements, device='cuda', dtype=torch.float32):
    num_nodes = len(nodes)
    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elem_t = torch.tensor(elements, dtype=torch.long, device=device)

    coords = nodes_t[elem_t]
    v1 = coords[:, 1] - coords[:, 0]
    v2 = coords[:, 2] - coords[:, 0]
    areas = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

    m_loc_template = torch.tensor([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=dtype, device=device) / 12.0

    m_locs = areas.unsqueeze(1).unsqueeze(2) * m_loc_template.unsqueeze(0)

    rows = elem_t.unsqueeze(2).expand(-1, -1, 3).flatten()
    cols = elem_t.unsqueeze(1).expand(-1, 3, -1).flatten()
    vals = m_locs.flatten()

    return build_direct_csr(rows, cols, vals, num_nodes, device)

def compute_stiffness_matrix_2d(nodes, elements, device='cuda', dtype=torch.float32):
    num_nodes = len(nodes)
    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elem_t = torch.tensor(elements, dtype=torch.long, device=device)

    coords = nodes_t[elem_t]
    v1 = coords[:, 1] - coords[:, 0]
    v2 = coords[:, 2] - coords[:, 0]
    areas = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

    b = torch.stack([
        coords[:, 1, 1] - coords[:, 2, 1],
        coords[:, 2, 1] - coords[:, 0, 1],
        coords[:, 0, 1] - coords[:, 1, 1]
    ], dim=1) / (2.0 * areas.unsqueeze(1))

    c = torch.stack([
        coords[:, 2, 0] - coords[:, 1, 0],
        coords[:, 0, 0] - coords[:, 2, 0],
        coords[:, 1, 0] - coords[:, 0, 0]
    ], dim=1) / (2.0 * areas.unsqueeze(1))

    s_locs = areas.unsqueeze(1).unsqueeze(2) * (
        torch.bmm(b.unsqueeze(2), b.unsqueeze(1)) + torch.bmm(c.unsqueeze(2), c.unsqueeze(1))
    )

    rows = elem_t.unsqueeze(2).expand(-1, -1, 3).flatten()
    cols = elem_t.unsqueeze(1).expand(-1, 3, -1).flatten()
    vals = s_locs.flatten()

    return build_direct_csr(rows, cols, vals, num_nodes, device)

def get_g_quadrature_2d(nodes, elements, device='cuda', dtype=torch.float32):
    # Highly accurate symmetric 12th-order quadrature rule for triangles
    num_elements = len(elements)
    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elem_t = torch.tensor(elements, dtype=torch.long, device=device)
    coords = nodes_t[elem_t]

    w = torch.tensor([0.116786275726379, 0.116786275726379, 0.116786275726379,
                      0.050844906370207, 0.050844906370207, 0.050844906370207,
                      0.082851075618374, 0.082851075618374, 0.082851075618374,
                      0.082851075618374, 0.082851075618374, 0.082851075618374], dtype=dtype, device=device)
    w /= torch.sum(w)

    alpha = torch.tensor([0.063089014491502, 0.873821971016996, 0.063089014491502,
                          0.249286745170910, 0.501426509658180, 0.249286745170910,
                          0.310352451033784, 0.053145049844816, 0.636502499121400,
                          0.636502499121400, 0.310352451033784, 0.053145049844816], dtype=dtype, device=device)
    beta = torch.tensor([0.063089014491502, 0.063089014491502, 0.873821971016996,
                         0.249286745170910, 0.249286745170910, 0.501426509658180,
                         0.053145049844816, 0.636502499121400, 0.310352451033784,
                         0.053145049844816, 0.636502499121400, 0.310352451033784], dtype=dtype, device=device)
    gamma = 1.0 - alpha - beta

    bary = torch.stack([alpha, beta, gamma], dim=1)
    quad_pts = torch.einsum('evd, qv -> eqd', coords, bary)

    qx = quad_pts[:, :, 0]
    qy = quad_pts[:, :, 1]

    pi = np.pi
    g_eval = torch.sin(2.0 * pi * qx) * torch.sin(2.0 * pi * qy)
    g_avg = torch.sum(g_eval * w.unsqueeze(0), dim=1)
    return g_avg

def get_grad_2d(nodes, elements, device='cuda', dtype=torch.float32):
    num_nodes = len(nodes)
    num_elements = len(elements)

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elem_t = torch.tensor(elements, dtype=torch.long, device=device)

    coords = nodes_t[elem_t]
    v1 = coords[:, 1] - coords[:, 0]
    v2 = coords[:, 2] - coords[:, 0]
    areas = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

    b = torch.stack([
        coords[:, 1, 1] - coords[:, 2, 1],
        coords[:, 2, 1] - coords[:, 0, 1],
        coords[:, 0, 1] - coords[:, 1, 1]
    ], dim=1) / (2.0 * areas.unsqueeze(1))

    c = torch.stack([
        coords[:, 2, 0] - coords[:, 1, 0],
        coords[:, 0, 0] - coords[:, 2, 0],
        coords[:, 1, 0] - coords[:, 0, 0]
    ], dim=1) / (2.0 * areas.unsqueeze(1))

    rows = torch.arange(num_elements, dtype=torch.long, device=device).unsqueeze(1).expand(-1, 3).flatten()
    cols = elem_t.flatten()

    indices = torch.stack([rows, cols])
    grad_x_t = torch.sparse_coo_tensor(indices, b.flatten(), size=(num_elements, num_nodes), device=device).coalesce().to_sparse_csr()
    grad_y_t = torch.sparse_coo_tensor(indices, c.flatten(), size=(num_elements, num_nodes), device=device).coalesce().to_sparse_csr()

    return grad_x_t, grad_y_t

@torch.jit.script
def update_p_vectors_compiled(px_temp, py_temp, px, py):
    p_norm = torch.maximum(torch.ones_like(px_temp), torch.sqrt(px_temp**2 + py_temp**2))
    torch.div(px_temp, p_norm, out=px)
    torch.div(py_temp, p_norm, out=py)

@torch.jit.script
def update_cg_vectors_compiled(x, p, r, Ap, alpha_cg):
    x.add_(p, alpha=alpha_cg)
    r.add_(Ap, alpha=-alpha_cg)

@torch.jit.script
def run_cg_iter_compiled(p, Ap, r, x, r_dot_old_ref, rtol_sq: float):
    r_dot_old = r_dot_old_ref[0]
    denom = torch.dot(p, Ap)
    if denom == 0:
        return False
    alpha_cg = r_dot_old / denom
    update_cg_vectors_compiled(x, p, r, Ap, alpha_cg)
    r_dot_new = torch.dot(r, r)
    if r_dot_new < rtol_sq:
        return False
    p.mul_(r_dot_new / r_dot_old).add_(r)
    r_dot_old_ref[0] = r_dot_new
    return True

def cg_solver_inplace(A_sparse, b, x, r, p, Ap, r_dot_old_ref, rtol=1e-4, max_iter=100):
    # Re-use pre-allocated buffers inplace to achieve zero dynamic memory footprint
    torch.mv(A_sparse, x, out=r)
    torch.sub(b, r, out=r)
    p.copy_(r)
    r_dot_old_ref[0] = torch.dot(r, r)
    rtol_sq = rtol ** 2

    for _ in range(max_iter):
        torch.mv(A_sparse, p, out=Ap)
        if not run_cg_iter_compiled(p, Ap, r, x, r_dot_old_ref, rtol_sq):
            break
    return x

def Solve_2D_ROF(nodes, elements, alpha, T, noise_level=0.0, eps_stop=1e-5, max_its=3000, device='cuda', dtype=torch.float32):
    with torch.no_grad():
        num_nodes = len(nodes)
        num_elements = len(elements)
        Nx = int(np.sqrt(num_nodes)) - 1
        h = 1.0 / Nx

        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elem_t = torch.tensor(elements, dtype=torch.long, device=device)

        coords = nodes_t[elem_t]
        v1 = coords[:, 1] - coords[:, 0]
        v2 = coords[:, 2] - coords[:, 0]
        areas_t = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

        M = compute_mass_matrix_2d(nodes, elements, device=device, dtype=dtype)
        S = compute_stiffness_matrix_2d(nodes, elements, device=device, dtype=dtype)
        grad_x, grad_y = get_grad_2d(nodes, elements, device=device, dtype=dtype)

        g = get_g_quadrature_2d(nodes, elements, device=device, dtype=dtype)

        C_g = torch.zeros(num_nodes, dtype=dtype, device=device)
        C_g.index_add_(0, elem_t.flatten(), (g * areas_t / 3.0).repeat_interleave(3))

        g_mean = torch.sum(g * areas_t) / torch.sum(areas_t)
        u = torch.full((num_nodes,), g_mean, dtype=dtype, device=device)
        du = torch.zeros(num_nodes, dtype=dtype, device=device)
        px = torch.zeros(num_elements, dtype=dtype, device=device)
        py = torch.zeros(num_elements, dtype=dtype, device=device)

        A = (M + S * h)
        lhs_matrix = (M + S * h + M * (alpha * T))

        residuals = []
        grad_x_T = torch.transpose(grad_x, 0, 1).to_sparse_csr()
        grad_y_T = torch.transpose(grad_y, 0, 1).to_sparse_csr()

        eps_stop_sq = eps_stop ** 2

        # Pre-allocate iteration buffers
        u_tilde = torch.empty_like(u)
        px_temp = torch.empty_like(px)
        py_temp = torch.empty_like(py)
        ux = torch.empty_like(px)
        uy = torch.empty_like(py)
        div_p = torch.empty_like(u)
        div_p_temp = torch.empty_like(u)
        rhs = torch.empty_like(u)
        mu_buf = torch.empty_like(u)

        # Pre-allocate CG workspace buffers to completely avoid dynamic runtime allocation
        cg_r = torch.empty_like(du)
        cg_p = torch.empty_like(du)
        cg_Ap = torch.empty_like(du)
        r_dot_old_ref = torch.tensor([0.0], dtype=dtype, device=device)

        for iteration in range(max_its):
            torch.add(u, du, alpha=T, out=u_tilde)

            torch.mv(grad_x, u_tilde, out=ux)
            torch.mv(grad_y, u_tilde, out=uy)

            torch.add(px, ux, alpha=T, out=px_temp)
            torch.add(py, uy, alpha=T, out=py_temp)

            update_p_vectors_compiled(px_temp, py_temp, px, py)

            torch.mv(grad_x_T, areas_t * px, out=div_p)
            torch.mv(grad_y_T, areas_t * py, out=div_p_temp)
            div_p.add_(div_p_temp).mul_(-1.0)

            torch.mv(M, u, out=mu_buf)
            torch.add(div_p, C_g, alpha=alpha, out=rhs)
            rhs.add_(mu_buf, alpha=-alpha)

            cg_solver_inplace(lhs_matrix, rhs, du, cg_r, cg_p, cg_Ap, r_dot_old_ref, rtol=1e-4)
            u.add_(du, alpha=T)

            res_sq = torch.dot(du, torch.mv(A, du))
            residuals.append(torch.sqrt(res_sq).item())

            if res_sq < eps_stop_sq:
                break

        return u.cpu().numpy(), px.cpu().numpy(), py.numpy() if device.type == 'cpu' else py.cpu().numpy(), residuals, g.cpu().numpy()

def calculate_2d_rof(nodes, elements, u_np, g_np, alpha, device='cuda', dtype=torch.float32):
    with torch.no_grad():
        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elem_t = torch.tensor(elements, dtype=torch.long, device=device)
        u = torch.tensor(u_np, dtype=dtype, device=device)
        g = torch.tensor(g_np, dtype=dtype, device=device)

        coords = nodes_t[elem_t]
        v1 = coords[:, 1] - coords[:, 0]
        v2 = coords[:, 2] - coords[:, 0]
        areas = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

        b = torch.stack([
            coords[:, 1, 1] - coords[:, 2, 1],
            coords[:, 2, 1] - coords[:, 0, 1],
            coords[:, 0, 1] - coords[:, 1, 1]
        ], dim=1) / (2.0 * areas.unsqueeze(1))

        c = torch.stack([
            coords[:, 2, 0] - coords[:, 1, 0],
            coords[:, 0, 0] - coords[:, 2, 0],
            coords[:, 1, 0] - coords[:, 0, 0]
        ], dim=1) / (2.0 * areas.unsqueeze(1))

        u_elem = u[elem_t]
        ux = torch.sum(b * u_elem, dim=1)
        uy = torch.sum(c * u_elem, dim=1)

        grad_magnitude = torch.sqrt(ux**2 + uy**2)
        tv_energy = torch.sum(grad_magnitude * areas).item()

        w = torch.tensor([0.116786275726379, 0.116786275726379, 0.116786275726379,
                          0.050844906370207, 0.050844906370207, 0.050844906370207,
                          0.082851075618374, 0.082851075618374, 0.082851075618374,
                          0.082851075618374, 0.082851075618374, 0.082851075618374], dtype=dtype, device=device)
        w /= torch.sum(w)

        alpha_coord = torch.tensor([0.063089014491502, 0.873821971016996, 0.063089014491502,
                                    0.249286745170910, 0.501426509658180, 0.249286745170910,
                                    0.310352451033784, 0.053145049844816, 0.636502499121400,
                                    0.636502499121400, 0.310352451033784, 0.053145049844816], dtype=dtype, device=device)
        beta_coord = torch.tensor([0.063089014491502, 0.063089014491502, 0.873821971016996,
                                   0.249286745170910, 0.249286745170910, 0.501426509658180,
                                   0.053145049844816, 0.636502499121400, 0.310352451033784,
                                   0.053145049844816, 0.636502499121400, 0.310352451033784], dtype=dtype, device=device)
        gamma_coord = 1.0 - alpha_coord - beta_coord

        bary = torch.stack([alpha_coord, beta_coord, gamma_coord], dim=1)

        quad_pts = torch.einsum('evd, qv -> eqd', coords, bary)
        qx = quad_pts[:, :, 0]
        qy = quad_pts[:, :, 1]

        pi = np.pi
        g_eval = torch.sin(2.0 * pi * qx) * torch.sin(2.0 * pi * qy)

        u_eval = torch.einsum('ev, qv -> eq', u_elem, bary)

        l2_diff_sq = (u_eval - g_eval) ** 2
        fidelity_energy = 0.5 * alpha * torch.sum(torch.sum(l2_diff_sq * w.unsqueeze(0), dim=1) * areas).item()

        total_energy = tv_energy + fidelity_energy
        return total_energy, tv_energy, fidelity_energy

# Run the optimized execution structure
N_values = [16 * i for i in range(1,50)]
energies_study = []
alpha_study = 100

start_time = time.time()
for N_val in N_values:
    h_val = 1.0 / N_val
    T_val = np.sqrt(h_val) / 4.0
    nodes_study, elements_study = generate_mesh_2d(N_val, N_val)

    u_res, _, _, _, g_res = Solve_2D_ROF(
        nodes_study, elements_study, alpha=alpha_study, T=T_val, noise_level=0.0, max_its=10000, eps_stop=1e-8, device=device, dtype=dtype
    )

    total_e, _, _ = calculate_2d_rof(nodes_study, elements_study, u_res, g_res, alpha=alpha_study, device=device, dtype=dtype)
    energies_study.append(total_e)
    print(f"Solution with N = {N_val} Completed Successfully: I = {total_e:.6f}")
print(f"All resolutions finished in {time.time() - start_time:.2f} seconds!")