from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


"""2D quasi-static AT2 phase-field fracture of a notched plate in tension.

Run: python PhaseFieldFacture/main.py
Dependencies: numpy, scipy, matplotlib.
Units: mm, N, MPa; plane strain, with force reported per chosen thickness.

Energy: integral[g(d)*psi0 + Gc/2*(d**2/ell + ell*|grad(d)|**2)].
d = 0: intact; d = 1: broken; g(d) = (1-k)*(1-d)**2 + k.
Linear triangular finite elements and alternating displacement/damage solves
are used. A history field H = max(psi0) enforces irreversible damage.
This simple isotropic model degrades both tension and compression: use it for
the tensile example below, not crack closure or compression-dominated failure.
"""


# 1. Parameters: edit these directly to change the simulation.
width = 1.0                 # Plate width, mm
height = 1.0                # Plate height, mm
thickness = 1.0             # Out-of-plane thickness, mm
nx = 81                     # Nodes along x
ny = 81                     # Nodes along y; odd so the crack lies on a node row
young_modulus = 210000.0    # E, MPa = N/mm^2
poisson_ratio = 0.3
fracture_energy = 2.7       # Gc, N/mm
length_scale = 0.04         # ell, mm; resolve this using several elements
residual_stiffness = 1e-7   # k, avoids a singular fully broken solid
initial_crack_length = 0.4  # Horizontal crack from the left edge, mm
max_displacement = 0.015    # Final vertical displacement of the top edge, mm
load_steps = 100
snapshot_every = 10
max_iterations = 1500       # Crack jumps can require many alternating solves
tolerance = 1e-5
output_dir = Path(__file__).resolve().parent / "output1"

if nx < 5 or ny < 5 or ny % 2 != 1:
    raise ValueError("Use nx >= 5 and an odd ny >= 5")
if min(width, height, thickness, young_modulus, fracture_energy, length_scale) <= 0:
    raise ValueError("Dimensions, E, Gc, and ell must be positive")
if not -1 < poisson_ratio < 0.5 or not 0 < residual_stiffness < 1:
    raise ValueError("Require -1 < nu < 0.5 and 0 < residual_stiffness < 1")
if not 0 < initial_crack_length < width:
    raise ValueError("The initial crack must end inside the plate")
if min(load_steps, snapshot_every, max_iterations) < 1 or tolerance <= 0:
    raise ValueError("Step counts and tolerance must be positive")
if max_displacement <= 0:
    raise ValueError("max_displacement must be positive for this tensile example")
if max(width / (nx - 1), height / (ny - 1)) > length_scale / 2:
    print("Warning: refine the mesh; aim for element spacing <= ell/2.")
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Structured mesh, split into counterclockwise linear triangles.
x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
xx, yy = np.meshgrid(x, y)
points = np.column_stack((xx.ravel(), yy.ravel()))
nodes = np.arange(nx * ny).reshape(ny, nx)
lower_left = nodes[:-1, :-1].ravel()
lower_right = nodes[:-1, 1:].ravel()
upper_left = nodes[1:, :-1].ravel()
upper_right = nodes[1:, 1:].ravel()
# Alternate diagonals to preserve reflection symmetry about the crack line.
# A single diagonal orientation can artificially steer a mode-I crack.
even_cell = np.indices((ny - 1, nx - 1)).sum(axis=0).ravel() % 2 == 0
triangles = np.vstack((
    np.column_stack((lower_left, lower_right, np.where(even_cell, upper_right, upper_left))),
    np.column_stack((np.where(even_cell, lower_left, lower_right), upper_right, upper_left)),
))
node_count = len(points)
element_count = len(triangles)
coords = points[triangles]
area = 0.5 * (
    (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1])
    - (coords[:, 2, 0] - coords[:, 0, 0]) * (coords[:, 1, 1] - coords[:, 0, 1])
)

# Shape-function gradients are constant inside each triangle.
grad = np.empty((element_count, 3, 2))
grad[:, :, 0] = coords[:, [1, 2, 0], 1] - coords[:, [2, 0, 1], 1]
grad[:, :, 1] = coords[:, [2, 0, 1], 0] - coords[:, [1, 2, 0], 0]
grad /= 2 * area[:, None, None]

# B maps the six element displacement values to [eps_xx, eps_yy, gamma_xy].
B = np.zeros((element_count, 3, 6))
B[:, 0, 0::2] = grad[:, :, 0]
B[:, 1, 1::2] = grad[:, :, 1]
B[:, 2, 0::2] = grad[:, :, 1]
B[:, 2, 1::2] = grad[:, :, 0]
nu = poisson_ratio
C = young_modulus / ((1 + nu) * (1 - 2 * nu)) * np.array([
    [1 - nu, nu, 0],
    [nu, 1 - nu, 0],
    [0, 0, (1 - 2 * nu) / 2],
])
element_stiffness = np.einsum("eai,ab,ebj,e->eij", B, C, B, area)
element_dofs = np.empty((element_count, 6), dtype=int)
element_dofs[:, 0::2] = 2 * triangles
element_dofs[:, 1::2] = 2 * triangles + 1
u_rows = np.repeat(element_dofs, 6, axis=1).ravel()
u_cols = np.tile(element_dofs, (1, 6)).ravel()
d_rows = np.repeat(triangles, 3, axis=1).ravel()
d_cols = np.tile(triangles, (1, 3)).ravel()
gradient_matrix = coo_matrix((
    (fracture_energy * length_scale
     * np.einsum("eik,ejk,e->eij", grad, grad, area)).ravel(),
    (d_rows, d_cols),
), shape=(node_count, node_count)).tocsc()
nodal_volume = np.bincount(
    triangles.ravel(), weights=np.repeat(area / 3, 3), minlength=node_count
)


# 3. Boundary conditions: uy=0 at bottom, uy=U at top; free lateral edges.
# Fix one horizontal displacement to remove rigid horizontal translation.
bottom = nodes[0]
top = nodes[-1]
fixed_u = np.concatenate((2 * bottom + 1, 2 * top + 1, [2 * bottom[0]]))
free_u = np.setdiff1d(np.arange(2 * node_count), fixed_u)
crack_nodes = nodes[ny // 2, x <= initial_crack_length]
free_d = np.setdiff1d(np.arange(node_count), crack_nodes)


def solve_displacement(damage, imposed_displacement):
    # Exact element average of g(d) for a linearly interpolated damage field.
    intact = 1 - damage[triangles]
    degradation = residual_stiffness + (1 - residual_stiffness) * (
        intact.sum(axis=1)**2 + (intact**2).sum(axis=1)
    ) / 12
    stiffness = coo_matrix((
        (degradation[:, None, None] * element_stiffness).ravel(),
        (u_rows, u_cols),
    ), shape=(2 * node_count, 2 * node_count)).tocsc()
    displacement = np.zeros(2 * node_count)
    displacement[2 * top + 1] = imposed_displacement
    displacement[free_u] = spsolve(
        stiffness[free_u][:, free_u], -(stiffness @ displacement)[free_u]
    )
    return displacement, stiffness


def solve_damage(history):
    # AT2: -Gc*ell*laplace(d) + (Gc/ell + 2*(1-k)*H)*d = 2*(1-k)*H.
    # Lump the reaction term onto nodes; outer edges have natural zero flux.
    driving_force = np.bincount(
        triangles.ravel(),
        weights=np.repeat(2 * (1 - residual_stiffness) * history * area / 3, 3),
        minlength=node_count,
    )
    matrix = gradient_matrix + diags(fracture_energy / length_scale * nodal_volume
                                    + driving_force, format="csc")
    damage = np.zeros(node_count)
    damage[crack_nodes] = 1
    damage[free_d] = spsolve(
        matrix[free_d][:, free_d], (driving_force - matrix @ damage)[free_d]
    )
    return np.clip(damage, 0, 1)


def save_snapshot(step, imposed_displacement):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    image = axes[0].tripcolor(
        points[:, 0], points[:, 1], triangles, damage,
        shading="gouraud", cmap="inferno", vmin=0, vmax=1,
    )
    axes[0].set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
                title="Damage: 0 = intact, 1 = broken")
    figure.colorbar(image, ax=axes[0], label="d")
    image = axes[1].tripcolor(
        points[:, 0], points[:, 1], triangles, displacement[1::2],
        shading="gouraud", cmap="viridis",
    )
    axes[1].set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
                title="Vertical displacement (mm)")
    figure.colorbar(image, ax=axes[1], label="uy (mm)")
    figure.suptitle(f"2D AT2 fracture | step {step}/{load_steps} | U = {imposed_displacement:.5f} mm")
    figure.savefig(output_dir / f"fracture_{step:04d}.png", dpi=150)
    plt.close(figure)
    np.savez_compressed(
        output_dir / f"fields_{step:04d}.npz", x=x, y=y,
        damage=damage.reshape(ny, nx),
        ux=displacement[0::2].reshape(ny, nx),
        uy=displacement[1::2].reshape(ny, nx),
        imposed_displacement=imposed_displacement,
    )


# 4. Quasi-static loading with alternating displacement and damage solves.
history = np.zeros(element_count)
damage = solve_damage(history)  # Equilibrated diffuse profile around the seed.
displacement = np.zeros(2 * node_count)
load_history = [[0.0, 0.0, 0.0, float(damage.mean()), 0]]
save_snapshot(0, 0.0)
start = perf_counter()
print(f"Mesh: {nx} x {ny} nodes, {element_count} triangles", flush=True)
print(f"Snapshots: {output_dir}", flush=True)

for step in range(1, load_steps + 1):
    imposed_displacement = max_displacement * step / load_steps
    for iteration in range(1, max_iterations + 1):
        old_damage = damage.copy()
        old_displacement = displacement.copy()
        displacement, stiffness = solve_displacement(damage, imposed_displacement)
        strain = np.einsum("eij,ej->ei", B, displacement[element_dofs])
        elastic_density = 0.5 * np.einsum("ei,ij,ej->e", strain, C, strain)
        history = np.maximum(history, elastic_density)
        # H never decreases. The nodal maximum also guards against roundoff healing.
        damage = np.maximum(old_damage, solve_damage(history))
        if not (np.isfinite(damage).all() and np.isfinite(displacement).all()):
            raise FloatingPointError(f"Non-finite solution at load step {step}")
        damage_error = np.max(np.abs(damage - old_damage))
        displacement_error = np.linalg.norm(displacement - old_displacement) / max(
            np.linalg.norm(displacement), 1e-14
        )
        if iteration % 100 == 0:
            print(f"  Step {step}, iteration {iteration}: "
                  f"error={max(damage_error, displacement_error):.2e}", flush=True)
        if max(damage_error, displacement_error) < tolerance:
            break
    else:
        raise RuntimeError(
            f"Step {step} did not converge after {max_iterations} iterations "
            f"(error={max(damage_error, displacement_error):.2e}). "
            "Increase load_steps or max_iterations."
        )

    # Restore mechanical equilibrium for the accepted damage before reporting forces.
    displacement, stiffness = solve_displacement(damage, imposed_displacement)
    reaction = thickness * (stiffness @ displacement)[2 * top + 1].sum()
    load_history.append([step, imposed_displacement, reaction, float(damage.mean()), iteration])
    np.savetxt(
        output_dir / "load_displacement.csv", load_history, delimiter=",", comments="",
        header="step,top_displacement_mm,reaction_N,mean_damage,iterations",
    )
    if step % snapshot_every == 0 or step == load_steps:
        save_snapshot(step, imposed_displacement)
    if step % 5 == 0 or step == 1 or iteration > 30:
        print(f"Step {step:3d}/{load_steps}: U={imposed_displacement:.5f} mm, "
              f"F={reaction:.3f} N, iterations={iteration}, "
              f"error={max(damage_error, displacement_error):.2e}", flush=True)


# 5. Save the full loading curve.
load_history = np.asarray(load_history)
figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
axis.plot(load_history[:, 1], load_history[:, 2], "o-", markersize=3)
axis.set(xlabel="Top displacement (mm)", ylabel="Top reaction (N)",
         title="Load-displacement curve")
axis.grid(alpha=0.3)
figure.savefig(output_dir / "load_displacement.png", dpi=150)
plt.close(figure)
print(f"Simulation complete in {perf_counter() - start:.1f} s. Results: {output_dir}")
