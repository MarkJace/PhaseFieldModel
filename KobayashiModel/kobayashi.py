import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

## Kobayashi model - stabilized version by Copilot

# dimension and time settings
Lx = 100
Ly = 100
dx = 1.0
dy = 1.0
dt = 1e-5
tsimu = 3000
toutput = 100

# parameters
tau = 1e-4          # evolution coeff   
aniso = 6           # anisotropy mode number
ang0 = 0.0          # initial orientation angle
eps = 0.01          # average interface energy
delta = 0.02        # anisotropy strength
alpha = 1.8         # supercooling coeff: alpha < 1
gamma = 50.0        # temperature enlarging coeff
K = 1.8             # dimensionless latent heat

# initialization
radius = 5
width = 1.2
p = np.zeros((Lx, Ly), dtype=float)
T = np.zeros((Lx, Ly), dtype=float)
Te = 1.0

# initial spherical nuclei with a smooth interface
cx, cy = Lx // 2, Ly // 2   # center of the n
x_coords = np.arange(Lx)
y_coords = np.arange(Ly)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
p = 0.5 * (1.0 - np.tanh((r - radius) / width))

# define the output directory
output_dir = Path.cwd() / "simu_results"
output_dir.mkdir(exist_ok=True)

# Simulation starts
print("Starting stabilized Kobayashi model simulation...")
print(f"Grid: {Lx}x{Ly}, Time steps: {tsimu}, dt = {dt}")
print(f"Output interval: {toutput}")
print(f"Output directory: {output_dir}")

for step in range(tsimu):
    # derivative of p respect to x: px (dp/dx)
    px = np.zeros_like(p)
    px[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dx)
    px[0, :] = (p[1, :] - p[0, :]) / dx
    px[-1, :] = (p[-1, :] - p[-2, :]) / dx

    # derivative of p respect to y: py (dp/dy)
    py = np.zeros_like(p)
    py[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dy)
    py[:, 0] = (p[:, 1] - p[:, 0]) / dy
    py[:, -1] = (p[:, -1] - p[:, -2]) / dy

    # position angle matrix (theta) and the anisotropy term (a_theta)
    theta = np.arctan2(Y - cy, X - cx)
    a_theta = 1.0 + delta * np.cos(aniso * (theta - ang0))
    a_sq = a_theta ** 2

    # a^2 * px and a^2 * py
    a2_px = a_sq * px
    a2_py = a_sq * py

    # derivatives of a^2 * grad p: a^2 *(dp/dx + dp/dy)
    div_a2_grad_p = np.zeros_like(p)
    div_a2_grad_p[1:-1, :] = (a2_px[2:, :] - a2_px[:-2, :]) / (2 * dx)
    div_a2_grad_p[0, :] = (a2_px[1, :] - a2_px[0, :]) / dx
    div_a2_grad_p[-1, :] = (a2_px[-1, :] - a2_px[-2, :]) / dx
    div_a2_grad_p[:, 1:-1] += (a2_py[:, 2:] - a2_py[:, :-2]) / (2 * dy)
    div_a2_grad_p[:, 0] += (a2_py[:, 1] - a2_py[:, 0]) / dy
    div_a2_grad_p[:, -1] += (a2_py[:, -1] - a2_py[:, -2]) / dy

    # anisotropy angular derivative epsilon' = d epsilon / d theta
    deps_dtheta = -delta * aniso * np.sin(aniso * (theta - ang0))
    # eps * eps'
    eps_eps_prime = a_theta * deps_dtheta

    # cross terms from Kobayashi's full anisotropic operator
    x_term = np.zeros_like(p)
    y_term = np.zeros_like(p)

    x_term[1:-1, :] = (eps_eps_prime[2:, :] * py[2:, :] - eps_eps_prime[:-2, :] * py[:-2, :]) / (2 * dx)
    x_term[0, :] = (eps_eps_prime[1, :] * py[1, :] - eps_eps_prime[0, :] * py[0, :]) / dx
    x_term[-1, :] = (eps_eps_prime[-1, :] * py[-1, :] - eps_eps_prime[-2, :] * py[-2, :]) / dx

    y_term[:, 1:-1] = (eps_eps_prime[:, 2:] * px[:, 2:] - eps_eps_prime[:, :-2] * px[:, :-2]) / (2 * dy)
    y_term[:, 0] = (eps_eps_prime[:, 1] * px[:, 1] - eps_eps_prime[:, 0] * px[:, 0]) / dy
    y_term[:, -1] = (eps_eps_prime[:, -1] * px[:, -1] - eps_eps_prime[:, -2] * px[:, -2]) / dy

    # term for the supercooling
    m = alpha * np.tanh(gamma * (Te - T))

    # rhs_phi: corresponding to the full Kobayashi phase-field anisotropic operator
    rhs_phi = eps ** 2 * (div_a2_grad_p ) - x_term + y_term + p * (1.0 - p) * (p - 0.5 + m)

    # phase field evolution
    p_new = p + (dt / tau) * rhs_phi
    p_new = np.clip(p_new, 0.0, 1.0)

    laplacian_T = (
        np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4 * T
    ) / (dx * dy)

    dphi_dt = (p_new - p) / dt
    T_new = T + dt * (laplacian_T + K * dphi_dt)

    T_new[0, :] = 0
    T_new[-1, :] = 0
    T_new[:, 0] = 0
    T_new[:, -1] = 0

    p = p_new
    T = T_new

    if step % toutput == 0:
        elapsed = step / tsimu * 100
        solid_fraction = np.mean(p > 0.5) * 100
        print(f"Step {step:6d}/{tsimu} ({elapsed:5.1f}%) | Solid fraction: {solid_fraction:5.2f}% | p range: [{p.min():.3f}, {p.max():.3f}]")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axes[0].imshow(p, cmap='hot', origin='lower', vmin=0, vmax=1)
        axes[0].set_title(f'Phase Field (Step {step})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(T, cmap='coolwarm', origin='lower')
        axes[1].set_title(f'Temperature (Step {step})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(output_dir / f'simu_step_{step:06d}.png', dpi=100)
        plt.close()

        print(f"  -> Saved {output_dir / f'simu_step_{step:06d}.png'}")

print("Simulation completed!")
