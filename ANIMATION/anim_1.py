import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
epsilon_0 = 1  # Assume normalized permittivity
q1, q2 = 1.0, -1.0  # Charge values (+ and -)
L = 5  # Grid size
N = 100  # Grid resolution
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Initial positions of the charges
x1_start, y1_start = -2, 0  # Charge 1 (left)
x2_start, y2_start = 2, 0   # Charge 2 (right)
x1_end, x2_end = -0.5, 0.5  # Final positions (closer together)

# Time steps for movement
num_frames = 50
x1_positions = np.linspace(x1_start, x1_end, num_frames)
x2_positions = np.linspace(x2_start, x2_end, num_frames)

# Function to compute potential using Green's function
def compute_potential(x1, y1, x2, y2):
    r1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
    r2 = np.sqrt((X - x2)**2 + (Y - y2)**2)
    
    r1[r1 == 0] = 1e-12  # Avoid singularity
    r2[r2 == 0] = 1e-12  # Avoid singularity

    # Compute the potential
    Phi = (q1 / (2 * np.pi * epsilon_0)) * np.log(1 / r1) + (q2 / (2 * np.pi * epsilon_0)) * np.log(1 / r2)
    return Phi

# Setup figure for animation
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, compute_potential(x1_start, y1_start, x2_start, y2_start), levels=50, cmap="RdBu_r")
charge1_plot, = ax.plot([], [], 'ro', markersize=10)  # Charge 1
charge2_plot, = ax.plot([], [], 'bo', markersize=10)  # Charge 2
fig.colorbar(contour, label="Potential")

def update(frame):
    x1, x2 = x1_positions[frame], x2_positions[frame]
    Phi = compute_potential(x1, y1_start, x2, y2_start)
    
    for c in contour.collections:
        c.remove()
    ax.contourf(X, Y, Phi, levels=50, cmap="RdBu_r")

    charge1_plot.set_data([x1], [y1_start])
    charge2_plot.set_data([x2], [y2_start])
    return contour, charge1_plot, charge2_plot

# Run animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
plt.title("Electrostatic Potential Evolution")
plt.show()
