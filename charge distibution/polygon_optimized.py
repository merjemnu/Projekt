import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

# Constants
epsilon_0 = 8.85e-12  # Vacuum permittivity (for electrostatics)

# Green's function for a 2D domain with Dirichlet boundary condition
def greens_function(x, y, x_prime, y_prime):
    """Green's function for 2D Laplace equation with Dirichlet boundary condition."""
    r = np.sqrt((x - x_prime)**2 + (y - y_prime)**2)
    # Handle the singularity at r = 0
    r[r == 0] = np.finfo(float).eps  # Replace 0 with a small number to avoid division by zero
    return -1/(2 * np.pi) * np.log(r)

# Define the geometry of the weird-looking object (arbitrary shape for demonstration)
def geometry(x, y):
    """Define a weird-looking object in the domain."""
    # Example: an irregular polygon (e.g., a triangle or a free-form shape)
    # You can define this shape via an array of vertices.
    # Here, a simple example of a polygon (say, a pentagon for simplicity).
    polygon_points = np.array([
        [0.3, 0.7],
        [0.7, 0.8],
        [0.9, 0.5],
        [0.5, 0.2],
        [0.2, 0.3]
    ])
    
    # Check if the point (x, y) is inside the polygon
    path = Path(polygon_points)
    points = np.vstack((x.flatten(), y.flatten())).T
    mask = path.contains_points(points)
    return mask.reshape(x.shape)

# Calculate the potential on a grid using vectorized operations
def calculate_potential(x_grid, y_grid):
    """Calculate the potential at each point in the domain using Green's function."""
    potential = np.zeros_like(x_grid)
    
    # Create a mask for points inside the geometry
    inside_mask = geometry(x_grid, y_grid)
    
    # Vectorized calculation of potential
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            if inside_mask[i, j]:  # Check if inside the object
                r = np.sqrt((x_grid - x_grid[i, j])**2 + (y_grid - y_grid[i, j])**2)
                r[r == 0] = np.finfo(float).eps  # Replace 0 with a small number to avoid division by zero
                potential[i, j] = np.sum(inside_mask * greens_function(x_grid[i, j], y_grid[i, j], x_grid, y_grid))
    
    # Apply Dirichlet boundary condition (set boundary potential to 0)
    potential[0, :] = 0  # Boundary at the bottom (y = -1)
    potential[-1, :] = 0  # Boundary at the top (y = 1)
    potential[:, 0] = 0  # Boundary at the left (x = -1)
    potential[:, -1] = 0  # Boundary at the right (x = 1)

    return potential

# Main function to visualize the object and potential
def main():
    # Define the grid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate the potential
    potential = calculate_potential(X, Y)

    # Plot the weird-looking object (using a polygon as an example)
    plt.figure(figsize=(10, 5))
    
    ax = plt.gca()
    polygon_points = np.array([
        [0.3, 0.7],
        [0.7, 0.8],
        [0.9, 0.5],
        [0.5, 0.2],
        [0.2, 0.3]
    ])
    polygon = Polygon(polygon_points, closed=True, edgecolor='r', facecolor='none', lw=2)
    ax.add_patch(polygon)
    
    # Plot the potential field
    plt.contourf(X, Y, potential, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    plt.title("Potential Field from 2D Charge Distribution on Irregular Object")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

# Ensure that the plot only runs when visualisation.py is executed directly
if __name__ == "__main__":
    main()