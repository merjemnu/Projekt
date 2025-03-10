import numpy as np
import matplotlib.pyplot as plt

# Define grid
N = 50
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Compute Green's function
rho = np.sqrt(X**2 + Y**2)
G = (1 / (2 * np.pi)) * np.log(rho)
G[rho == 0] = 0  # Avoid singularity at the source

# Define source term
f = -6  # Constant source term

# Compute convolution integral (discretized version)
dx = x[1] - x[0]
dy = y[1] - y[0]
solution = dx * dy * f * G

# Apply boundary condition (analytical form)
u_boundary = 1 + X**2 + 2 * Y**2
solution += u_boundary

# Plot the solution
plt.contourf(X, Y, solution, levels=50, cmap="viridis")
plt.colorbar(label="u(x,y)")
plt.title("Solution of Poisson’s Equation Using Green’s Function")
plt.show()