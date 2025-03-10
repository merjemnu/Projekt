from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
from petsc4py import PETSc

# Erstelle ein Gitter
L = 1.0  # Domänenlänge
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[-L/2, -L/2], [L/2, L/2]], [50, 50], cell_type=mesh.CellType.triangle)

# Definiere den Funktionsraum
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# Definiere Test- und Ansatzfunktionen
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Punktquelle als Delta-ähnliche Funktion
x0, y0 = 0.0, 0.0  # Position der Punktquelle
delta = fem.Function(V)
dofs = fem.locate_dofs_geometrical(V, lambda x: np.sqrt((x[0] - x0)**2 + (x[1] - y0)**2) < 0.05)
delta.x.array[dofs] = 1.0

# Schwache Formulierung
f = delta  # Rechte Seite mit Punktquelle
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Dirichlet-Randbedingungen (u=0 am Rand)
boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: np.abs(x[0]) == L/2 or np.abs(x[1]) == L/2)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

# Löse das Problem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Visualisierung
import pyvista
from dolfinx.plot import create_vtk_mesh

grid = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
grid.point_data["uh FEM"] = uh.x.array
if domain.comm.rank == 0:
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, scalars="uh FEM", cmap="viridis")
    plotter.show()
