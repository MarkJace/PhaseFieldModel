# PhaseFieldModel
A examples repository for phase field model

* Cahn-Hillard: Cahn-Hillard model for spinodal decomposition
* Grain Growth: Allen-Cahn model for Voronoi structured polycrystal evolution
* Kobayashi model: Kobayashi model for dendrite growth under temperature control



## Cahh-Hillard
Here a Cahn-Hillard model is applied to simulate the spinodal decomposition.
In the model, the free energy functional is
$$F[c] = ∫_V [ A·c²(1-c)² + (κ/2)·|∇c|² ] dV$$
The Cahn-Hillard equation:
$$∂c/∂t = ∇·(M∇μ) = M·∇²[f'(c)] - Mκ·∇⁴c$$
Here the equation is solved by spectrum method.

## Grain Growth
Here a phase field model is achieved to simulate the grain growth for the polycrystal case.
In the model, Allen-Cahn equation is applied
$$dη_i/dt = L_i [ A·η_i - B·η_i^3 - 2η_i·Σ_{j≠i} η_j^2 + κ_i·∇²η_i ]$$
The initial structure is obtained by Voronoi structure.

# Kobayashi model
Here a Kobayashi model is applied to simulate the dendrite growth.
The model refers to the paper "Modeling and numerical simulations of dendritic crystal growth"(https://doi.org/10.1016/0167-2789(93)90120-P).
The dentrite growth process and temperature evolution are achieved by the program.




