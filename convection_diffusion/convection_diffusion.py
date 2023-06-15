"""Script to analyse the dispersion coefficient with DarSIA and data fitting."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spsp

xaxis = np.linspace(0, 3, 100)
time = 1
D = 1e-2
velocity = 1
porosity = 1

# Analytical solution
analytical_solution = 0.5 * spsp.erfc(
    (porosity * xaxis - velocity * time) / (2 * (D * porosity * time) ** 0.5)
) + 0.5 * np.exp(velocity * xaxis / D) * spsp.erfc(
    (porosity * xaxis + velocity * time) / (2 * (D * porosity * time) ** 0.5)
)

plt.plot(xaxis, analytical_solution)
plt.show()
