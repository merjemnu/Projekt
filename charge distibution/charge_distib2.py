import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon_0 = 8.85e-12  # Vacuum permittivity (for electrostatics)

# Green's function for a 2D domain (Dirichlet condition at boundary)
def greens_function(x, y, x_prime, y_prime):
    """Green's function for 2D Laplace equation with Dirichlet boundary condition."""
    r = np.sqrt((x - x_prime)**2 + (y - y_prime)**2)
    # Handle the singularity at r = 0
    r[r == 0] = np.finfo(float).eps  # Replace 0 with a small number to avoid division by zero
    return -1/(2 * np.pi) * np.log(r)

# Define the charge distribution (a circular distribution in this case)
def charge_distribution(x, y):
    """Define the 2D charge distribution."""
    # Example: circular charge distribution with radius 0.2 centered at (0, 0)
    radius = 0.2
    return np.exp(-((x**2 + y**2) / (2 * radius**2)))  # Gaussian-like distribution

# Calculate the potential on a grid using vectorized operations
def calculate_potential(x_grid, y_grid, charge_dist):
    """Calculate the potential at each point in the domain using Green's function."""
    potential = np.zeros_like(x_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            r = np.sqrt((x_grid - x_grid[i, j])**2 + (y_grid - y_grid[i, j])**2)
            r[r == 0] = np.finfo(float).eps  # Replace 0 with a small number to avoid division by zero
            potential[i, j] = np.sum(charge_dist * (-1/(2 * np.pi) * np.log(r)))
    
    return potential

# Define the grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Calculate the charge distribution
charge_dist = charge_distribution(X, Y)

# Plot the charge distribution
plt.figure()
plt.contourf(X, Y, charge_dist, levels=50, cmap='viridis')
plt.colorbar(label="Charge Distribution")
plt.title("Charge Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Calculate the potential
potential = calculate_potential(X, Y, charge_dist)

# Plot the potential
plt.figure()
plt.contourf(X, Y, potential, levels=50, cmap='viridis')
plt.colorbar(label="Potential")
plt.title("Potential Field from Charge Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()