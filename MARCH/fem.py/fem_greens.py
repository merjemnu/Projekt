import numpy as np
import matplotlib.pyplot as plt
from femm import fem_poisson_solver

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
G[rho == 0] = 0  # Avoid singularity at the source

# FEM solution (Replace with your actual FEM solution)
U_FEM = np.loadtxt("MARCH/femm.py")  # Load FEM result from a file (Modify this as needed)

# Compute error function
error = np.abs(G - U_FEM)

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Green's function solution (Analytical)
ax[0].contourf(X, Y, G, levels=50, cmap="viridis")
ax[0].set_title("Analytical Solution (Green's Function)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_aspect("equal")

# Plot FEM solution (Numerical)
ax[1].contourf(X, Y, U_FEM, levels=50, cmap="plasma")
ax[1].set_title("Numerical Solution (FEM)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_aspect("equal")

# Plot error function
ax[2].contourf(X, Y, error, levels=50, cmap="inferno")
ax[2].set_title("Error |Green's - FEM|")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].set_aspect("equal")

# Show plots
plt.colorbar(ax[0].collections[0], ax=ax[0], label="u(x,y)")
plt.colorbar(ax[1].collections[0], ax=ax[1], label="u(x,y)")
plt.colorbar(ax[2].collections[0], ax=ax[2], label="Error")

plt.tight_layout()
plt.show()
