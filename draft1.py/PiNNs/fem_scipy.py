import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Define grid
N = 50  # Number of nodes per dimension
L = 1  # Length of the domain
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
nodes = np.array([[x[i], y[j]] for i in range(N) for j in range(N)])

# Create a mesh (triangles)
def get_elements(N):
    elements = []
    for i in range(N - 1):
        for j in range(N - 1):
            idx = i * N + j
            elements.append([idx, idx + 1, idx + N])  # First triangle
            elements.append([idx + 1, idx + N + 1, idx + N])  # Second triangle
    return elements

elements = get_elements(N)

# Stiffness matrix and load vector
K = lil_matrix((N*N, N*N))  # Stiffness matrix
F = np.zeros(N*N)  # Load vector

# Function to compute the element stiffness matrix (for linear triangles)
def element_stiffness(i, j, k):
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    x3, y3 = nodes[k]
    
    # Area of the triangle
    A = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    # Element stiffness matrix
    B = np.array([[y2 - y3, y3 - y1, y1 - y2], 
                  [x3 - x2, x1 - x3, x2 - x1]]) / (2*A)
    
    Ke = A * np.dot(B.T, B)
    return Ke

# Assemble global stiffness matrix and load vector
for element in elements:
    Ke = element_stiffness(element[0], element[1], element[2])
    for i in range(3):
        for j in range(3):
            K[element[i], element[j]] += Ke[i, j]

# Apply source term f(x, y)
f = np.full(N*N, -6)  # Constant source term
F += f

# Apply boundary conditions (Dirichlet: u = 1 on boundaries)
boundary_nodes = np.array([i for i in range(N*N) if nodes[i][0] == 0 or nodes[i][1] == 0])
for node in boundary_nodes:
    K.rows[node] = [node]
    K.data[node] = [1]
    F[node] = 1

# Solve the linear system
u = spsolve(K.tocsc(), F)

# Reshape the solution back to a 2D grid
u_solution = u.reshape((N, N))

# Plot the solution
plt.contourf(X, Y, u_solution, levels=50, cmap="viridis")
plt.colorbar(label="u(x, y)")
plt.title("Solution of Poisson's Equation Using FEM")
plt.show()
