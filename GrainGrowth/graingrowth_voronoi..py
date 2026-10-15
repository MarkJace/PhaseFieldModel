import numpy as np
import matplotlib.pyplot as plt
import os
import time


'''
Grain Growth — Voronoi

Final format of Allen-Cahn equation:
dη_i/dt = L_i [ A·η_i - B·η_i^3 - 2η_i·Σ_{j≠i} η_j^2 + κ_i·∇²η_i ]
'''


# ============================================================
# 1. Simulation settings
# ============================================================
# system para
Nx = 128          # grid number of x direction
Ny = 128          # grid number of y direction
dx = 0.5          # x grid size
dy = 0.5          # y grid size

# physical para
N_grain = 20      # grain number
A  = 1.0          # coefficient for bulk free energy A 
B  = 1.0          # coefficient for bulk free energy A 
L  = 1.0          # mobility
kappa = 1.0       # coefficient for gradient energy

# time evolu para
nstep  = 5000     # total step number
nprint = 500      # output once every nprint steps
dtime  = 0.01     # step time length

# ============================================================
# 2. Initial state: Voronoi structure
# ============================================================
def generate_voronoi_initial(Nx, Ny, N_grain, interface_width=3.0):
    """
    Generate the initial Voronoi structure

    mechanism:
      1. random create N_grain seed points (cx_i, cy_i)
      2. calculate the distance between point (x,y) and seed, d_i
      3. use softmax / tanh to create a fluent order parameter field η_i(x,y)

    Para:
      Nx, Ny: mesh size
      N_grain: grain number
      interface_width: grain boundary width (grid number)

    Return:
      eta: shape (N_grain, Nx, Ny)
      seeds: shape (N_grain, 2)
    """
    # 2.1 random seed points (avoid the seed position is too close to system boundary)
    margin = 5
    seeds_x = np.random.uniform(margin, Nx - margin, N_grain)
    seeds_y = np.random.uniform(margin, Ny - margin, N_grain)

    # 2.2 create the mesh grid position array 
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    # 2.3 initialize the order parameter fields
    eta = np.zeros((N_grain, Nx, Ny))

    # 2.4 calculate the nearest seed for every mesh points
    for i in range(Nx):
        for j in range(Ny):
            dist = np.sqrt((i - seeds_x)**2 + (j - seeds_y)**2)

            # find the nearest seed (idx1) and the second nearest seed (idx2)
            idx_sorted = np.argsort(dist)
            idx1, idx2 = idx_sorted[0], idx_sorted[1]
            d1, d2 = dist[idx1], dist[idx2]

            # construct the fluent interface by tanh function
            phi = (d2 - d1) / interface_width
            eta[idx1, i, j] = 0.5 * (1.0 + np.tanh(phi))

    # 2.5 normalization: Σ η_i² ≈ 1
    norm = np.sqrt(np.sum(eta**2, axis=0)) + 1e-12
    eta /= norm

    return eta, np.column_stack([seeds_x, seeds_y])


# 2.6 another method to create the initial Vonoroi strcture by vectors (for big mesh)
def generate_voronoi_initial_fast(Nx, Ny, N_grain, interface_width=3.0):
    margin = 5
    seeds_x = np.random.uniform(margin, Nx - margin, N_grain)
    seeds_y = np.random.uniform(margin, Ny - margin, N_grain)

    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')  # (Nx, Ny)

    # calculate the distance between grid point and seed: shape (N_grain, Nx, Ny)
    dist = np.sqrt((xx[None, :, :] - seeds_x[:, None, None])**2 +
                   (yy[None, :, :] - seeds_y[:, None, None])**2)

    # sort it and obtain the nearest and the second nearest seed for every grid point
    idx_sorted = np.argsort(dist, axis=0)  # (N_grain, Nx, Ny)
    idx1 = idx_sorted[0]   # the nearest
    idx2 = idx_sorted[1]   # the second nearest

    d1 = np.take_along_axis(dist, idx1[None, :, :], axis=0)[0]  # (Nx, Ny)
    d2 = np.take_along_axis(dist, idx2[None, :, :], axis=0)[0]  # (Nx, Ny)

    # calculate the order para field
    phi = (d2 - d1) / interface_width
    phi_vals = 0.5 * (1.0 + np.tanh(phi))  # (Nx, Ny)

    # corresponding to grains
    eta = np.zeros((N_grain, Nx, Ny))
    for n in range(N_grain):
        eta[n] = np.where(idx1 == n, phi_vals, 0.0)

    # normalize
    norm = np.sqrt(np.sum(eta**2, axis=0)) + 1e-12
    eta /= norm

    return eta, np.column_stack([seeds_x, seeds_y])


# ============================================================
# 3. output and save
# ============================================================
outdir = 'output1'
os.makedirs(outdir, exist_ok=True)

def save_grain_map(eta, step, outdir, seeds=None):
    """save the results to outdir/"""
    grain_map = np.argmax(eta, axis=0)          # (Nx, Ny)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grain_map.T, origin='lower', cmap='tab20',
                   interpolation='none', vmin=0, vmax=N_grain-1)
    if seeds is not None:
        ax.scatter(seeds[:, 0], seeds[:, 1], c='white', marker='x', s=40, linewidths=1.5)
    ax.set_title(f'Step {step}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, ticks=range(0, N_grain, max(1, N_grain//10)))
    fname = f'{outdir}/step_{step:05d}.png'
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    return fname


# ============================================================
# 4. Construct initial Voronoi structure
# ============================================================
eta, seeds = generate_voronoi_initial_fast(Nx, Ny, N_grain, interface_width=3.0)
print(f"phase field shape: {eta.shape}  (N_grain={N_grain}, Nx={Nx}, Ny={Ny})")

# save initial state
save_grain_map(eta, 0, outdir, seeds)
print(f"Initial state saved at {outdir}/step_00000.png")

# # ============================================================
# # 4b. Interface test
# # ============================================================
# # y = Ny//2 line, x - eta curve
# row = Ny // 2
# fig, ax = plt.subplots(figsize=(10, 5))
# for n in range(min(N_grain, 8)):
#     ax.plot(eta[n, :, row], lw=1.5, label=f'$\\eta_{{{n}}}$')
# ax.set_xlabel('x (grid)')
# ax.set_ylabel(r'$\eta_i$')
# ax.set_title(f'Order parameters along row y={row} (interface width ≈ 2×3 grid)')
# ax.legend(ncol=4, fontsize=8)
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{outdir}/interface_profile.png', dpi=150)
# plt.close()
# print(f"interface test fig saved to {outdir}/interface_profile.png")

# ============================================================
# 5. Spectrum method — semi-implicit Allen-Cahn evolution
# ============================================================
# Mechanism:
#   ∂η/∂t = Lκ ∇²η + NL(η)   
#   where NL = L·(Aη - Bη³ - 2η Σ_{j≠i} η_j²)
#   in spectrum space: ∂η̂/∂t = -Lκ k² η̂ + NL̂

# ----- 5a. vector wave k² -----
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, dx)       # (Nx,)
ky = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)       # (Ny,)
KX, KY = np.meshgrid(kx, ky, indexing='ij')      # (Nx, Ny)
k2 = KX**2 + KY**2                                # (Nx, Ny)
denom_factor = 1.0 + dtime * L * kappa * k2       

# ----- 5b. define a function to compute the nonlinear term -----
def compute_nonlinear(eta):
    """
    calculate NL = L * (A*η - B*η³ - 2*η * Σ_{j≠i} η_j²)
    eta: (N_grain, Nx, Ny)
    return: NL, shape (N_grain, Nx, Ny)
    """
    sum_eta_sq = np.sum(eta**2, axis=0)  # Σ_j η_j², (Nx, Ny)
    NL = np.zeros_like(eta)
    for i in range(N_grain):
        eta_i = eta[i]
        # Σ_{j≠i} η_j² = Σ_all η_j² - η_i²
        cross_term = 2.0 * eta_i * (sum_eta_sq - eta_i**2)
        NL[i] = L * (A * eta_i - B * eta_i**3 - cross_term)
    return NL

# ----- 5c. calculation run a step in spectrum space -----
def step_forward(eta_hat, NL_hat):
    eta_hat_new = (eta_hat + dtime * NL_hat) / denom_factor
    return eta_hat_new

# ----- 5d. main evolution loop -----
print(f"\nEvolution starts: nstep={nstep}, dtime={dtime}, nprint={nprint}")
print(f"Spectrum method | mesh: {Nx}×{Ny} | grain number: {N_grain}")


# initialize the spectrum space
eta_hat = np.fft.fft2(eta, axes=(1, 2))
# start to count the time
t_start = time.time()

for step in range(1, nstep + 1):
    # nonlinear term: NL
    NL = compute_nonlinear(eta)
    NL_hat = np.fft.fft2(NL, axes=(1, 2))

    # calculation in spectrum space
    eta_hat = step_forward(eta_hat, NL_hat)

    # use iFT to convert the real space to the spectrum space and remove the imaginary part
    eta = np.fft.ifft2(eta_hat, axes=(1, 2)).real

    # cut off all numbers out of [0, 1]
    eta = np.clip(eta, 0.0, 1.0)

    # output
    if step % nprint == 0:
        grain_map = np.argmax(eta, axis=0)
        n_grains_remaining = len(np.unique(grain_map))
        elapsed = time.time() - t_start
        print(f"  Step {step:5d}/{nstep} | "
              f"remained grain number: {n_grains_remaining:2d} | "
              f"used time: {elapsed:.1f}s")
        save_grain_map(eta, step, outdir)

# ----- 5e. save the result -----
save_grain_map(eta, nstep, outdir)
elapsed_total = time.time() - t_start
print(f"\nSimulation done! totally used time: {elapsed_total:.1f}s")
print(f"results saved to {outdir}/")



