# Cahn-Hilliard model
## Introduction
This script main.py is applied to simulate the spinodal decomposition, the process of compositional decomposition and evolution.

## Model
The simulation seetings and parameters refers to the Part1: Simulation settings.

There is only one composition in the simulation.
In a 2D simulation frame, there is an initial concentration with a random noise and the mean concentration(fraction) is 0.5 (50%).

The free energy function of th system is the sum of bulk energy and interfacial energy:

$$F[c] = ∫_V [ A·c²(1-c)² + (κ/2)·|∇c|² ] dV$$

Based on the free energy functional, chemical potential is obtained:

$$μ = δF/δc = f'(c) - κ·∇²c$$

where $$f'(c) = 2A·c(1-c)(1-2c) = 2A(c - 3c² + 2c³)$$


Cahn-Hilliard equation (Governing Equation) is:

$$∂c/∂t = ∇·(M∇μ) = M·∇²[f'(c)] - Mκ·∇⁴c$$

The spectral method is used to solving the governing equation.

## Program

The program relies on numpy and matplotlib.

To start the simulation, run:
``` python
python main.py
```
The shapshots are generated in the directory output1.