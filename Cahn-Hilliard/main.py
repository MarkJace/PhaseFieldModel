import numpy as np
import matplotlib.pyplot as plt
import time
import os


'''
Cahn-Hilliard Equation — spinodal decomposition (spectral method)

Free energy functional:
    F[c] = ∫_V [ A·c²(1-c)² + (κ/2)·|∇c|² ] dV

Chemical potential (variational derivative):
    μ = δF/δc = f'(c) - κ·∇²c ,   f'(c) = 2A·c(1-c)(1-2c) = 2A(c - 3c² + 2c³)

Cahn-Hilliard equation (conserved field c, np.fft.fftfreq(8)          # [0, .125, .25, .375, -.5, -.375, -.25, -.125]
np.fft.fftfreq(8, d=0.5)   # [0, .25, .5, .75, -1., -.75, -.5, -.25]constant mobility M):
    ∂c/∂t = ∇·(M∇μ) = M·∇²[f'(c)] - Mκ·∇⁴c
    (note: the RHS still needs an outer ∇² acting on μ)

Spectral semi-implicit scheme:
    split f'(c) = s·c + g(c),  g(c) = f'(c) - s·c,  s = 2A (stabilization)
    in Fourier space:  ∇² → -k²,  ∇⁴ → k⁴
    ∂ĉ/∂t = -Mk²(s + κk²)·ĉ - Mk²·ĝ    (linear part implicit, nonlinear explicit)
    ĉⁿ⁺¹ = [ (1 + Δt·M·s·k²)·ĉⁿ - Δt·M·k²·ĝⁿ ] / [ 1 + Δt·M·k²·(s + κk²) ]
'''


# ============================================================
# 1. Simulation settings
# ============================================================
# system para
Nx = 128          # grid number of x direction
Ny = 128          # grid number of y direction
dx = 1.0          # x grid size
dy = 1.0          # y grid size

# physical para
A     = 1.0       # coefficient of the double-well potential
kappa = 0.5       # coefficient of the gradient energy
M     = 1.0       # mobility


# initial state para
c_mean    = 0.5    # average concentration (inside the spinodal region)
noise_amp = 0.02   # initial fluctuation amplitude
rng_seed  = 42     # random seed (reproducible)

# time evolution para
nstep  = 3000     # total step number
nprint = 100      # output once every nprint steps
dtime  = 0.05     # time step length

# ============================================================
# 2. Wave number arrays in Fourier space
# ============================================================
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, dx)     # (Nx,)
ky = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)     # (Ny,)
KX, KY = np.meshgrid(kx, ky, indexing='ij')   # (Nx, Ny)
k2 = KX**2 + KY**2                            # |k|²
k4 = k2**2                                    # |k|⁴

# linear-splitting stabilization coefficient: s >= max|f''(c)| = 2A
s = 2.0 * A
# denominator of the semi-implicit scheme (independent of c, computed only once)
denom = 1.0 + dtime * M * k2 * (s + kappa * k2)
numer_lin = 1.0 + dtime * M * s * k2          # linear part of the numerator

# ============================================================
# 3. Output and save
# ============================================================
outdir = 'output1'
os.makedirs(outdir, exist_ok=True)


def save_concentration(c, step, outdir):
    """save the concentration field c(x, y) as png"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c.T, origin='lower', cmap='RdBu_r', vmin=0.0, vmax=1.0,
                   interpolation='none')
    ax.set_title(f'Step {step}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='c')
    fname = f'{outdir}/step_{step:05d}.png'
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    return fname


# ============================================================
# 4. Initial state: uniform field + small random fluctuation
# ============================================================
rng = np.random.default_rng(rng_seed)
c = c_mean + noise_amp * rng.standard_normal((Nx, Ny))

# theoretical fastest-growing wavelength (linear stability analysis)
fpp = 2.0 * A * (1.0 - 6.0 * c_mean + 6.0 * c_mean**2)   # f''(c_mean)
k_m = np.sqrt(-fpp / (2.0 * kappa))
print(f"f''(c_mean) = {fpp:.3f}  ({'< 0, spinodal unstable' if fpp < 0 else '> 0, stable'})")
print(f"fastest-growing wavenumber k_m = {k_m:.3f}, wavelength λ_m = {2*np.pi/k_m:.2f} grids")

save_concentration(c, 0, outdir)
print(f"Initial state saved at {outdir}/step_00000.png")

# ============================================================
# 5. Spectral semi-implicit evolution
# ============================================================
def chemical_potential_deriv(c):
    """f'(c) = 2A·c(1-c)(1-2c)"""
    return 2.0 * A * c * (1.0 - c) * (1.0 - 2.0 * c)


def step_forward(c_hat, g_hat):
    """one step in Fourier space: ĉⁿ⁺¹ = (numer_lin·ĉⁿ - Δt·M·k²·ĝⁿ) / denom"""
    return (numer_lin * c_hat - dtime * M * k2 * g_hat) / denom


def total_free_energy(c, c_hat):
    """F = ∫ [ A·c²(1-c)² + (κ/2)·|∇c|² ] dV"""
    f_bulk = A * c**2 * (1.0 - c)**2
    # gradient via spectral differentiation: ∇c = IFFT( i·k·ĉ )
    grad_x = np.fft.ifft2(1j * KX * c_hat).real
    grad_y = np.fft.ifft2(1j * KY * c_hat).real
    f_grad = 0.5 * kappa * (grad_x**2 + grad_y**2)
    return np.sum(f_bulk + f_grad) * dx * dy


print(f"\nEvolution starts: nstep={nstep}, dtime={dtime}, nprint={nprint}")
print(f"Spectral method (semi-implicit, s={s}) | mesh: {Nx}×{Ny} | c_mean={c_mean}")

c_hat = np.fft.fft2(c)          # initial state in Fourier space
energy_list = []                # record the total free energy
t_start = time.time()

for step in range(1, nstep + 1):
    # nonlinear part: g(c) = f'(c) - s·c   (explicit)
    fp = chemical_potential_deriv(c)
    g = fp - s * c
    g_hat = np.fft.fft2(g)

    # update in Fourier space (k=0 mode: denom=1, ĝ=0 → mass exactly conserved)
    c_hat = step_forward(c_hat, g_hat)

    # inverse FT back to real space
    c = np.fft.ifft2(c_hat).real

    # output
    if step % nprint == 0:
        elapsed = time.time() - t_start
        energy = total_free_energy(c, c_hat)
        energy_list.append(energy)
        print(f"  Step {step:5d}/{nstep} | c ∈ [{c.min():.3f}, {c.max():.3f}] | "
              f"mean c = {c.mean():.5f} | F = {energy:.4f} | used time: {elapsed:.1f}s")
        save_concentration(c, step, outdir)

# ----- 5b. final results -----
save_concentration(c, nstep, outdir)
energy = total_free_energy(c, c_hat)
energy_list.append(energy)
elapsed_total = time.time() - t_start

# free energy vs time
steps_recorded = list(range(0, nstep + 1, nprint))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.array(steps_recorded) * dtime, energy_list, 'o-', lw=1.5)
ax.set_xlabel('time t'); ax.set_ylabel('total free energy F')
ax.set_title('Cahn-Hilliard: free energy decay')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{outdir}/free_energy.png', dpi=150); plt.close()
print(f"free energy curve saved to {outdir}/free_energy.png")

print(f"\nSimulation done! totally used time: {elapsed_total:.1f}s")
print(f"results saved to {outdir}/")












