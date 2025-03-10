from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh
from dolfinx.fem import functionspace
import dolfinx.fem as fem
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type

# Define the domain and function space for FEM
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
V = functionspace(domain, ("Lagrange", 1))

# Dirichlet boundary condition function
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# Define trial and test functions for FEM
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define source term (right-hand side of Poisson's equation)
f = fem.Constant(domain, default_scalar_type(-6))

# Define the bilinear form (a) and linear form (L) for FEM
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solve the linear problem using PETSc
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Define exact solution for comparison
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

# Compute error (L2 and max) between the exact and numerical solutions
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = np.max(np.abs(uD.x.array - uh.x.array))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Green's function solution for Poisson's equation in 2D
def greens_function_2d(X, Y, sources, strengths):
    potential = np.zeros_like(X)
    for (x0, y0), strength in zip(sources, strengths):
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        # Avoid division by zero at the source location
        r = np.maximum(r, 1e-12)
        # Green's function contribution
        potential += strength * np.log(1 / r) / (2 * np.pi)
    return potential

# Assuming uh.x.array is computed correctly and has the correct size
# For demonstration, let's create a dummy uh.x.array with the correct size
uh_x_array = np.random.rand(100, 100)  # Replace this with the actual computation


import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define sources and strengths
sources = [(0, 0)]
strengths = [1]

# Green's function for 2D Poisson
def greens_function_2d(X, Y, sources, strengths):
    potential = np.zeros_like(X)
    for (sx, sy), strength in zip(sources, strengths):
        r = np.sqrt((X - sx)**2 + (Y - sy)**2)
        potential += strength / (2 * np.pi * np.maximum(r, 1e-12))
    return potential

# Compute the exact solution
exact_solution = greens_function_2d(X, Y, sources, strengths)

# Assuming uh.x.array is computed correctly and has the correct size
# For demonstration, let's create a dummy uh.x.array with the correct size
uh_x_array = np.random.rand(100, 100)  # Replace this with the actual computation

# Plot the exact solution and numerical solution
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Exact solution plot
c1 = ax[0].contourf(X, Y, exact_solution, 50, cmap='viridis')
fig.colorbar(c1, ax=ax[0])
ax[0].set_title("Exact Solution")

# Numerical solution plot
c2 = ax[1].contourf(X, Y, uh_x_array, 50, cmap='viridis')  # Ensure uh_x_array has the correct size
fig.colorbar(c2, ax=ax[1])
ax[1].set_title("Numerical Solution")

plt.show()