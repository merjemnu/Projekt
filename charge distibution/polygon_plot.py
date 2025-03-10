import numpy as np
import matplotlib.pyplot as plt

x =np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

polygon_points = np.array([
        [0.3, 0.7],
        [0.7, 0.8],
        [0.9, 0.5],
        [0.5, 0.2],
        [0.2, 0.3]
    ])

    # Extract X and Y coordinates
poly_x = polygon_points[:, 0]
poly_y = polygon_points[:, 1]


 # Plot the polygon
 # Plot the polygon
plt.figure(figsize=(10, 5))
plt.fill(poly_x, poly_y, color='lightblue', edgecolor='black', linewidth=2, alpha=0.6)  # Fill the polygon

    # Plot points for better visualization
plt.scatter(poly_x, poly_y, color='red', zorder=3, label="Vertices")

    # Connect the last point to the first to close the shape
plt.plot(np.append(poly_x, poly_x[0]), np.append(poly_y, poly_y[0]), color='black', linewidth=2)

    # Labels and grid
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Polygon Plot")
plt.grid(True)
plt.legend()
plt.show()
