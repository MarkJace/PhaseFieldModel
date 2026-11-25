# PhaseFieldModel
A examples repository for phase field model.

Here is the index of the repository. The detailed information is included in the directories of the cases (document.md).

## For this repository
The programs are achieved by python3. The needed modules is in requirements.txt




## Cahh-Hillard
Cahn-Hillard model in Phase field simulation.
It simulates the compositional evolution (conserved fields).
In the model, the free energy functional is
$$F[c] = ∫_V [ A·c²(1-c)² + (κ/2)·|∇c|² ] dV$$
The Cahn-Hillard equation:
$$∂c/∂t = ∇·(M∇μ) = M·∇²[f'(c)] - Mκ·∇⁴c$$
Here the equation is solved by spectrum method.

## Ferroelectricity in Polycrystal



## Ferroelectricity in Single Crystal
In a stress-free, uniaxially polarized ferroelectric single crystal,








## Grain Growth
Here a phase field model is achieved to simulate the grain growth for the polycrystal case.
In the model, Allen-Cahn equation is applied
$$dη_i/dt = L_i [ A·η_i - B·η_i^3 - 2η_i·Σ_{j≠i} η_j^2 + κ_i·∇²η_i ]$$
The initial structure is obtained by Voronoi structure.

# Kobayashi Model
Here a Kobayashi model is applied to simulate the dendrite growth.
The model refers to the paper "Modeling and numerical simulations of dendritic crystal growth"(https://doi.org/10.1016/0167-2789(93)90120-P).
The dentrite growth process and temperature evolution are achieved by the program.

## Li Ion Battery Dendrite formation






# Phase Field Crystal Model
The script simulates the formation of an ordered phase from a mother phase.


# Phase Field Fracture Model
Here the fracture case is simulated by Phase Field Facture Model.
The script simulates a 2D material block with an initial crack.






