import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Define point sources and their strengths
sources = [(0, 0), (2, 2)]  # Source points
strengths = [1, -0.5]       # Source strengths

# Green's function solution for Poisson's equation
def greens_function_2d(X, Y, sources, strengths):
    potential = np.zeros_like(X)
    for (x0, y0), strength in zip(sources, strengths):
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        # Avoid division by zero at the source location
        r = np.maximum(r, 1e-12)
        # Green's function contribution
        potential += strength * np.log(1 / r) / (2 * np.pi)
    return potential

# Compute the potential
potential = greens_function_2d(X, Y, sources, strengths)

# Plot the potential
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, potential, levels=50, cmap='coolwarm')
plt.colorbar(contour)
plt.title("Potential Solved via Green's Function")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(*zip(*sources), color='black', marker='o', label='Sources')
plt.legend()
plt.show()
