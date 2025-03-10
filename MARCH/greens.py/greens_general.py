import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)


# Define the Laplacian operator for Poisson equation
def laplacian(X, Y):
    # Finite difference approximation for the Laplacian
    dX = X[1] - X[0]  # Grid spacing in X
    dY = Y[1] - Y[0]  # Grid spacing in Y
    d2X = np.diff(X, 2) / dX**2  # Second derivative in X
    d2Y = np.diff(Y, 2) / dY**2  # Second derivative in Y
    return d2X + d2Y 

# Define Green's function for general elliptical PDE (2D Laplace example)
def greens_function(X, Y, x0, y0):
    # Compute distance r between points (X, Y) and the source (x0, y0)
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    # Avoid division by zero at the source
    r = np.maximum(r, 1e-12)
    
    # Green's function for Laplace equation (Poisson equation)
    return 1 / (2 * np.pi) * np.log(1 / r)

# Example: Define a point source at (x0, y0)
x0, y0 = 0, 2  # Example point source location

# Calculate Green's function for the point source
G = greens_function(X, Y, x0, y0)

# Plot the Green's function (potential due to the point source)
plt.contourf(X, Y, G, levels=50, cmap='viridis')
plt.colorbar(label='Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Green's Function Potential (Point Source)")
plt.show()

# Define the source term for Poisson equation (e.g., point source strength)
source_strength = 1  # Strength of the point source

# Apply the Green's function to solve Poisson equation
def solve_poisson(X, Y, source_strength, x0, y0):
    G = greens_function(X, Y, x0, y0)  # Green's function for the source
    potential = source_strength * G  # Solution to Poisson equation using Green's function
    return potential

# Calculate the potential from the source
potential = solve_poisson(X, Y, source_strength, x0, y0)

# Plot the potential
plt.contourf(X, Y, potential, levels=50, cmap='viridis')
plt.colorbar(label='Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Potential from Point Source (Poisson Equation)")
plt.show()
