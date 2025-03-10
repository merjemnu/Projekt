import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

# Constants
epsilon_0 = 8.85e-12  # Vacuum permittivity (for electrostatics)

# Green's function for a 2D domain
def greens_function(x, y, x_prime, y_prime):
    """Green's function for the 2D Laplace equation."""
    r = np.sqrt((x - x_prime)**2 + (y - y_prime)**2)
    r[r == 0] = np.finfo(float).eps  # Avoid division by zero
    return -1 / (2 * np.pi) * np.log(r)

# Define the geometry of the object
def geometry(x, y):
    """Define an irregular polygonal object."""
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

# Calculate the potential on a grid
def calculate_potential(x_grid, y_grid):
    """Calculate the potential at each point using Green's function."""
    potential = np.zeros_like(x_grid)
    inside_mask = geometry(x_grid, y_grid)  # Define charge distribution
    
    # Vectorized calculation of potential
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            r = np.sqrt((x_grid - x_grid[i, j])**2 + (y_grid - y_grid[i, j])**2)
            r[r == 0] = np.finfo(float).eps  # Avoid singularity
            potential[i, j] = np.sum(inside_mask * greens_function(x_grid[i, j], y_grid[i, j], x_grid, y_grid))
    
    return potential

# Main function to visualize the potential
def main():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    potential = calculate_potential(X, Y)

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
    
    plt.contourf(X, Y, potential, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    plt.title("Potential Field from 2D Charge Distribution on Irregular Object")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

if __name__ == "__main__":
    main()
