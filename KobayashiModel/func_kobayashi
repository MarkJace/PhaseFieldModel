# func_kobayashi.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def kobayashi_simulation(
    # parameters
    tsimu=3000,
    toutput=100,
    tau=1e-4,
    aniso=6,
    ang0=0.0,
    eps=0.01,
    delta=0.02,
    alpha=0.8,
    gamma=50.0,
    K=1.8,
    # on grids
    Lx=100,
    Ly=100,
    dx=1.0,
    dy=1.0,
    dt=1e-5,
    # on initial nucleus
    radius=5,
    width=1.2,
    Te = 1,
    # boundary temperautre
    boundary_temp=0.0,
    # output directory
    output_dir=None
):
    """
    Kobayashi Model Parameters
    
    PARAMETERS:
        tsimu (int): total simu steps
        toutput (int): output steps
        tau (float): evolution coeff
        aniso (int): anisotropy mode number
        ang0 (float): initial orientation angle
        eps (float): average interfaceenergy
        delta (float): anisotropy strength
        alpha (float): supercooling coupling coeff
        gamma (float): temperature enlarging coeff
        K (float): dimensionless latent heat
        Lx, Ly (int): simu system length in x, y direction
        dx, dy (float): grid length in x, y direction
        dt (float): time step
        radius (float): initial nucleus radius
        width (float): interfacial width
        boundary_temp (float): boundary conditions for temperature
        output_dir (str/Path): output path
    """
    
    # initialization
    p = np.zeros((Lx, Ly), dtype=float)
    T = np.zeros((Lx, Ly), dtype=float)
    
    # initial nucleus
    cx, cy = Lx // 2, Ly // 2
    x_coords = np.arange(Lx)
    y_coords = np.arange(Ly)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    p = 0.5 * (1.0 - np.tanh((r - radius) / width))
    
    # output path
    if output_dir is None:
        output_dir = Path.cwd() / "simu_results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # print the simu info
    print("Starting Kobayashi model simulation...")
    print(f"Grid: {Lx}x{Ly}, Time steps: {tsimu}, dt = {dt}")
    print(f"Parameters: tau={tau}, aniso={aniso}, eps={eps}, delta={delta}")
    print(f"            alpha={alpha}, gamma={gamma}, K={K}")
    print(f"Output interval: {toutput}")
    print(f"Output directory: {output_dir}")
    
    # simu
    for step in range(tsimu):
        # px and py
        px = np.zeros_like(p)
        px[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dx)
        px[0, :] = (p[1, :] - p[0, :]) / dx
        px[-1, :] = (p[-1, :] - p[-2, :]) / dx
        
        py = np.zeros_like(p)
        py[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dy)
        py[:, 0] = (p[:, 1] - p[:, 0]) / dy
        py[:, -1] = (p[:, -1] - p[:, -2]) / dy
        
        # anisotropy function for interface energy
        theta = np.arctan2(Y - cy, X - cx)
        a_theta = 1.0 + delta * np.cos(aniso * (theta - ang0))
        a_sq = a_theta ** 2
        
        # derivative of the anisotropy terms
        a2_px = a_sq * px
        a2_py = a_sq * py
        
        div_a2_grad_p = np.zeros_like(p)
        div_a2_grad_p[1:-1, :] = (a2_px[2:, :] - a2_px[:-2, :]) / (2 * dx)
        div_a2_grad_p[0, :] = (a2_px[1, :] - a2_px[0, :]) / dx
        div_a2_grad_p[-1, :] = (a2_px[-1, :] - a2_px[-2, :]) / dx
        div_a2_grad_p[:, 1:-1] += (a2_py[:, 2:] - a2_py[:, :-2]) / (2 * dy)
        div_a2_grad_p[:, 0] += (a2_py[:, 1] - a2_py[:, 0]) / dy
        div_a2_grad_p[:, -1] += (a2_py[:, -1] - a2_py[:, -2]) / dy
        
        # cross interfacial gradient terms
        deps_dtheta = -delta * aniso * np.sin(aniso * (theta - ang0))
        eps_eps_prime = a_theta * deps_dtheta
        
        x_term = np.zeros_like(p)
        y_term = np.zeros_like(p)
        
        x_term[1:-1, :] = (eps_eps_prime[2:, :] * py[2:, :] - 
                          eps_eps_prime[:-2, :] * py[:-2, :]) / (2 * dx)
        x_term[0, :] = (eps_eps_prime[1, :] * py[1, :] - 
                       eps_eps_prime[0, :] * py[0, :]) / dx
        x_term[-1, :] = (eps_eps_prime[-1, :] * py[-1, :] - 
                        eps_eps_prime[-2, :] * py[-2, :]) / dx
        
        y_term[:, 1:-1] = (eps_eps_prime[:, 2:] * px[:, 2:] - 
                          eps_eps_prime[:, :-2] * px[:, :-2]) / (2 * dy)
        y_term[:, 0] = (eps_eps_prime[:, 1] * px[:, 1] - 
                       eps_eps_prime[:, 0] * px[:, 0]) / dy
        y_term[:, -1] = (eps_eps_prime[:, -1] * px[:, -1] - 
                        eps_eps_prime[:, -2] * px[:, -2]) / dy
        
        # supercooling term
        m = alpha * np.tanh(gamma * (Te - T))
        
        # right-hand term for the equation
        rhs_phi = (eps ** 2 * div_a2_grad_p - x_term + y_term + 
                   p * (1.0 - p) * (p - 0.5 + m))
        
        # phase field evolution
        p_new = p + (dt / tau) * rhs_phi
        p_new = np.clip(p_new, 0.0, 1.0)
        
        # tempearture field evolution
        laplacian_T = (
            np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
            np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4 * T
        ) / (dx * dy)
        
        dphi_dt = (p_new - p) / dt
        T_new = T + dt * (laplacian_T + K * dphi_dt)
        
        # boundary conditions
        T_new[0, :] = boundary_temp
        T_new[-1, :] = boundary_temp
        T_new[:, 0] = boundary_temp
        T_new[:, -1] = boundary_temp
        
        p = p_new
        T = T_new
        
        # output
        if step % toutput == 0:
            elapsed = step / tsimu * 100
            solid_fraction = np.mean(p > 0.5) * 100
            print(f"Step {step:6d}/{tsimu} ({elapsed:5.1f}%) | "
                  f"Solid fraction: {solid_fraction:5.2f}% | "
                  f"p range: [{p.min():.3f}, {p.max():.3f}]")
            
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
    return p, T


# run
if __name__ == "__main__":

    p_final, T_final = kobayashi_simulation()
