import numpy as np
import matplotlib.pyplot as plt

# Define grid
N = 100  # Grid size
L = 1  # Domain length
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Define point source location
x0, y0 = 0, 0  # Source at the center
q = 10

# Compute Green's function solution
rho = np.sqrt((X - x0)**2 + (Y - y0)**2)
G = (q / (2 * np.pi)) * np.log(rho)
G[rho == 0] = 0  # Avoid singularity (at source)

# Plot the solution
plt.contourf(X, Y, G, levels=50, cmap="viridis")
plt.colorbar(label="u(x,y)")
plt.title("Solution of Poisson’s Equation for a Point Source")
plt.show()

