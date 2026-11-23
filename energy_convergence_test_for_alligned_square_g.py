
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

def get_g_2d(nodes, elements, noise_level=0.0):
    num_elements = len(elements)
    g_face = np.zeros(num_elements, dtype=np.float32)
    centroids = np.mean(nodes[elements], axis=1)
    inside = (0.375 <= centroids[:, 0]) & (centroids[:, 0] <= 0.625) & (0.375 <= centroids[:, 1]) & (centroids[:, 1] <= 0.625)
    g_face[inside] = 1.0
    if noise_level > 0.0:
        g_face += np.random.normal(0, noise_level, num_elements).astype(np.float32)
    return g_face

g_test = get_g_2d(nodes, elements, noise_level=0.0)
print(f"Signal g initialized with {np.sum(g_test > 0)} triangles inside the square out of {len(g_test)} total triangles.")

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

# JIT Compilation to avoid graph overhead and enable lightweight CPU/GPU execution without compilation delays
@torch.jit.script
def update_p_vectors_compiled(px_temp, py_temp, px, py):
    p_norm = torch.maximum(torch.ones_like(px_temp), torch.sqrt(px_temp**2 + py_temp**2))
    torch.div(px_temp, p_norm, out=px)
    torch.div(py_temp, p_norm, out=py)

@torch.jit.script
def update_cg_vectors_compiled(x, p, r, Ap, alpha_cg):
    x.add_(p, alpha=alpha_cg)
    r.add_(Ap, alpha=-alpha_cg)

def Solve_2D_ROF(nodes, elements, alpha, T, noise_level=0.0, eps_stop=1e-8, max_its=10000, device='cuda', dtype=torch.float32):
    with torch.no_grad(): # Disable gradient tracking overhead completely
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

        g_cpu = get_g_2d(nodes, elements, noise_level=noise_level)
        g = torch.tensor(g_cpu, dtype=dtype, device=device)

        C_g = torch.zeros(num_nodes, dtype=dtype, device=device)
        C_g.index_add_(0, elem_t.flatten(), (g * areas_t / 3.0).repeat_interleave(3))

        g_mean = torch.sum(g * areas_t) / torch.sum(areas_t)
        u = torch.full((num_nodes,), g_mean, dtype=dtype, device=device)
        du = torch.zeros(num_nodes, dtype=dtype, device=device)
        px = torch.zeros(num_elements, dtype=dtype, device=device)
        py = torch.zeros(num_elements, dtype=dtype, device=device)

        A = (M + S * h)
        lhs_matrix = (M + S * h + M * (alpha * T))

        # Highly optimized in-place CG Solver
        def cg_solver(A_sparse, b, x0, rtol=1e-4, max_iter=100):
            x = x0.clone()
            r = b - torch.mv(A_sparse, x)
            p = r.clone()
            r_dot_old = torch.dot(r, r)
            rtol_sq = rtol ** 2

            for _ in range(max_iter):
                Ap = torch.mv(A_sparse, p)
                denom = torch.dot(p, Ap)
                if denom == 0:
                    break
                alpha_cg = r_dot_old / denom
                update_cg_vectors_compiled(x, p, r, Ap, alpha_cg)
                r_dot_new = torch.dot(r, r)
                if r_dot_new < rtol_sq:
                    break
                p.mul_(r_dot_new / r_dot_old).add_(r)
                r_dot_old = r_dot_new
            return x

        residuals = []
        grad_x_T = torch.transpose(grad_x, 0, 1).to_sparse_csr()
        grad_y_T = torch.transpose(grad_y, 0, 1).to_sparse_csr()

        eps_stop_sq = eps_stop ** 2

        # Pre-allocated workspace tensors
        u_tilde = torch.empty_like(u)
        px_temp = torch.empty_like(px)
        py_temp = torch.empty_like(py)

        for iteration in range(max_its):
            torch.add(u, du, alpha=T, out=u_tilde)

            ux = torch.mv(grad_x, u_tilde)
            uy = torch.mv(grad_y, u_tilde)

            torch.add(px, ux, alpha=T, out=px_temp)
            torch.add(py, uy, alpha=T, out=py_temp)

            update_p_vectors_compiled(px_temp, py_temp, px, py)

            div_p = -(torch.mv(grad_x_T, areas_t * px) + torch.mv(grad_y_T, areas_t * py))

            rhs = div_p - alpha * torch.mv(M, u) + alpha * C_g
            du = cg_solver(lhs_matrix, rhs, du, rtol=1e-4)
            u.add_(du, alpha=T)

            res_sq = torch.dot(du, torch.mv(A, du))
            residuals.append(torch.sqrt(res_sq).item())

            if res_sq < eps_stop_sq:
                break

        return u.cpu().numpy(), px.cpu().numpy(), py.numpy() if device.type == 'cpu' else py.cpu().numpy(), residuals, g_cpu

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

        u_cell_avg = torch.mean(u_elem, dim=1)
        fidelity_energy = 0.5 * alpha * torch.sum(((u_cell_avg - g)**2) * areas).item()

        total_energy = tv_energy + fidelity_energy
        return total_energy, tv_energy, fidelity_energy

# Run the optimized execution structure
N_values = [32, 48, 64, 96, 128, 144, 160, 176, 192, 224, 256, 288, 320, 352, 384, 512, 640, 768, 896, 1024, 1024 + 256, 1024 + 256*2, 1024 + 256*3, 1024 + 256*4]
energies_study = []
alpha_study = 100

start_time = time.time()
for N_val in N_values:
    h_val = 1.0 / N_val
    T_val = np.sqrt(h_val) / 4.0
    nodes_study, elements_study = generate_mesh_2d(N_val, N_val)

    u_res, _, _, _, g_res = Solve_2D_ROF(
        nodes_study, elements_study, alpha=alpha_study, T=T_val, noise_level=0.0, max_its=10000, eps_stop=1e-12, device=device, dtype=dtype
    )

    total_e, _, _ = calculate_2d_rof(nodes_study, elements_study, u_res, g_res, alpha=alpha_study, device=device, dtype=dtype)
    energies_study.append(total_e)
    print(f"Solution with N = {N_val} Completed Successfully: I = {total_e:.6f}")
print(f"All resolutions finished in {time.time() - start_time:.2f} seconds!")