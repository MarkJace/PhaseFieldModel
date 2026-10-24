from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without opening a GUI window.
import numpy as np
import matplotlib.pyplot as plt

""" 
a 2D Phase Field Crystal Model with periodic boundary conditions.

The governing equation:
    d(psi)/dt = laplacian([r + (1 + laplacian)^2] psi + psi^3)

the linear terms are treated implicitly in Fourier space
the cubic nonlinear term is treated explicitly.
"""


# 1. Parameters
grid_size = 128       # Number of grid points in each direction
dx = 1.0             # Grid spacing
dt = 0.1             # Time step
steps = 3000         # Number of time steps
r = -0.25            # Undercooling parameter
mean_density = -0.3  # Average density
noise = 0.01         # Amplitude of initial random noise
seed = 42            # Random seed for reproducibility
snapshot_count = 10  # Include the initial and final states
output_dir = Path(__file__).resolve().parent / "output1"

if grid_size < 4:
    raise ValueError("grid_size must be at least 4")
if dx <= 0 or dt <= 0:
    raise ValueError("dx and dt must be positive")
if steps < snapshot_count - 1:
    raise ValueError("steps must be at least snapshot_count - 1 for distinct snapshots")
if noise < 0:
    raise ValueError("noise cannot be negative")


# 2. Initial density field: average density + zero-mean random perturbation
rng = np.random.default_rng(seed)
initial_noise = rng.standard_normal((grid_size, grid_size))
initial_noise -= initial_noise.mean()
psi = mean_density + noise * initial_noise


# 3. Wave numbers and linear operator in Fourier space
# Under periodic boundary conditions, the laplacian corresponds to -k_squared.
wave_numbers = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=dx)
kx, ky = np.meshgrid(wave_numbers, wave_numbers, indexing="xy")
k_squared = kx**2 + ky**2
linear_operator = r + (1.0 - k_squared) ** 2
denominator = 1.0 + dt * k_squared * linear_operator

if np.any(denominator <= 0):
    raise ValueError(
        "dt is too large for the chosen parameters; "
        "the semi-implicit update denominator is not positive"
    )


# 4. Snapshot output at evenly spaced steps, including t = 0 and the final time
output_dir.mkdir(parents=True, exist_ok=True)
snapshot_steps = np.linspace(0, steps, snapshot_count, dtype=int)
snapshot_indices = {int(step): index for index, step in enumerate(snapshot_steps, start=1)}


def save_snapshot(step):
    time = step * dt
    output_path = output_dir / (
        f"pfc_{snapshot_indices[step]:02d}_step_{step:06d}_t_{time:g}.png"
    )
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(
        psi,
        origin="lower",
        extent=(0, grid_size * dx, 0, grid_size * dx),
        cmap="RdBu_r",
        interpolation="nearest",
    )
    axis.set(
        xlabel="x",
        ylabel="y",
        title=f"2D Phase-Field Crystal (t = {time:g})",
        aspect="equal",
    )
    figure.colorbar(image, ax=axis, label="density deviation")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved snapshot {snapshot_indices[step]}/{snapshot_count}: {output_path}")


# 5. Time evolution using semi-implicit Fourier spectral method
psi_hat = np.fft.fft2(psi)
conserved_zero_mode = psi_hat[0, 0]
save_snapshot(0)

for step in range(1, steps + 1):
    # Rebuild from the real field to suppress accumulated imaginary roundoff.
    psi_hat = np.fft.fft2(psi)
    cubic_hat = np.fft.fft2(psi**3)
    psi_hat = (psi_hat - dt * k_squared * cubic_hat) / denominator
    psi_hat[0, 0] = conserved_zero_mode  # Conserve the zero mode (mean density) in Fourier space
    psi = np.fft.ifft2(psi_hat).real
    if not np.isfinite(psi).all():
        raise FloatingPointError(f"Non-finite density at step {step}; try reducing dt")
    if step in snapshot_indices:
        save_snapshot(step)

print(
    f"Simulation complete: {grid_size}x{grid_size}, "
    f"{steps} steps; saved {snapshot_count} snapshots to {output_dir}"
)
print(f"Mean density: {psi.mean():.8f}")
