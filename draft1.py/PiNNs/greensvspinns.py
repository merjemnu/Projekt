import numpy as np
import matplotlib.pyplot as plt

# Define grid
N = 50
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Define parameters
M = 10  # Number of terms in the sum
solution = np.zeros_like(X)

# Compute Green's function solution
for m in range(1, M+1):
    for n in range(1, M+1):
        coef = (4 / (np.pi**2 * m * n)) * (-6)  # Source term factor
        factor = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        integral_term = (1 - (-1)**m) * (1 - (-1)**n) / (m * n * np.pi**2)
        solution += coef * factor * integral_term

# Add boundary condition (analytical form)
u_boundary = 1 + X**2 + 2 * Y**2
solution += u_boundary

# Plot the solution
plt.contourf(X, Y, solution, levels=50, cmap="viridis")
plt.colorbar(label="u(x,y)")
plt.title("Solution of Poisson’s Equation Using Green’s Function")
plt.show()
