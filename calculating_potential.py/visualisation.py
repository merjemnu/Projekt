import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon_0 = 8.85e-12  # Vacuum permittivity (for electrostatics)

# Green's function for a 2D domain (Dirichlet condition at boundary)
def greens_function(x, y, x_prime, y_prime):
    """Green's function for 2D Laplace equation with Dirichlet boundary condition."""
    r = np.sqrt((x - x_prime)**2 + (y - y_prime)**2)
    # Handle the singularity at r = 0
    if r == 0:
        return 0
    return -1/(2 * np.pi) * np.log(r)

# Define the geometry of the domain
def domain(x, y):
    """Define the 2D domain and obstacles."""
    # Example: circular obstacle at the center
    radius = 0.2
    return (x**2 + y**2 > radius**2)  # Inside the circle is an obstacle

# Calculate the potential on a grid
def calculate_potential(x_grid, y_grid, x_charge, y_charge):
    """Calculate the potential at each point in the domain using Green's function."""
    potential = np.zeros_like(x_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            if domain(x_grid[i, j], y_grid[i, j]):  # Only calculate inside the domain
                potential[i, j] = greens_function(x_grid[i, j], y_grid[i, j], x_charge, y_charge)
    
    return potential

# Main function to visualize the potential
def main():
    # Define the grid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Define the charge location
    x_charge, y_charge = 0.5, 0.5

    # Calculate the potential
    potential = calculate_potential(X, Y, x_charge, y_charge)

    # Plot the potential
    plt.contourf(X, Y, potential, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    plt.title("Potential Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Ensure that the plot only runs when visualisation.py is executed directly
if __name__ == "__main__":
    main()
