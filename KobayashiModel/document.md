# Kobayashi model
## Introduction
This script main.py is applied to simulate the dendrite growth under the temperature field during the solidification (Kobayashi model).

The paper describes the model: "Modeling and numerical simulations of dendritic crystal growth"(https://doi.org/10.1016/0167-2789(93)90120-P).

## Model
A 2D simulation frame is constructed. An initial grain seed is placed in the geometrical center of the simulation frame. 
Two field variables are considered in the model: the order parameter $\phi$ and dimensionless temperature $T$.


For the system, the free energy function is
$$F = \int_V \left[ \frac{1}{2}\epsilon(\theta)^2 |\nabla \phi|^2 + f(\phi, T) \right] dV$$

where, the bulk free energy density is 
$$ f(\phi, T) = \frac{1}{4}\phi^4
+ \left(\frac{1}{2} - \frac{1}{3}m\right)\phi^3
+ \left(\frac{1}{4} - \frac{1}{2}m\right)\phi^2 $$

where $m$ is 
$$m(T) = \frac{\alpha}{\pi} \tan^{-1}[\gamma(T_e - T)]$$

Based on the above equations, the governing equations are

For the phase field:
$$\tau \frac{\partial \phi}{\partial t} =
\nabla \cdot \big(\epsilon^2 \nabla \phi\big)
+ \partial_x\big( \epsilon \,\epsilon' \,\partial_y \phi \big)
- \partial_y\big( \epsilon \,\epsilon' \,\partial_x \phi \big)
+ \phi(1-\phi)\left(\phi - \tfrac{1}{2} + m\right)$$

For the temperature field:
$$\frac{\partial T}{\partial t} = \nabla^2 T + K\,\frac{\partial \phi}{\partial t}$$

Variational method is applied to solve the equations.

## Program

To start the simulation, run:
``` python
python main.py
```
The shapshots are generated in the directory output1.

