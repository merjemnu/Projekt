import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from dolfinx import mesh, fem
import ufl
from mpi4py import MPI

# Erzeuge ein 2D-Gitter für die FEM-Berechnung
nx, ny = 50, 50  # Anzahl der Elemente
msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [nx, ny], mesh.CellType.triangle)

# Funktionraum definieren

#V = fem.FunctionSpace(msh, ('CG', 1))  # P1-Lagrange-Elemente
V = fem.functionspace(msh, ("Lagrange", 1))  # Korrekte Syntax

# Green's Function Definition
def greens_function_2d(x, y, sources, strengths):
    G = np.zeros_like(x)
    for (sx, sy), strength in zip(sources, strengths):
        r = np.sqrt((x - sx)**2 + (y - sy)**2) + 1e-8  # Vermeidung von Singularität
        G += strength * np.log(r) / (2 * np.pi)
    return G

# Quellenpositionen und Stärken
sources = [(0.5, 0.5)]  # Punktquelle in der Mitte
strengths = [1.0]

# Definiere Variablen für FEM-Formulierung
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Function(V)

# Setze die Quellterme an den FEM-Knoten
coords = V.tabulate_dof_coordinates()
source_term = np.zeros(len(coords))
for i, (cx, cy) in enumerate(coords):
    for (sx, sy), strength in zip(sources, strengths):
        if np.linalg.norm([cx - sx, cy - sy]) < 0.02:  # Annäherung der Punktquelle
            source_term[i] = strength

f.x.array[:] = source_term

# Schwache Formulierung: -Δu = f
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Randbedingungen (Dirichlet u = 0 am Rand)
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0) | np.isclose(x[0], 1), np.isclose(x[1], 0) | np.isclose(x[1], 1))
bc = fem.dirichletbc(0.0, fem.locate_dofs_geometrical(V, boundary), V)

# Problem lösen
uh = fem.Function(V)
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], u=uh)
uh = problem.solve()

# Interpolation auf reguläres Gitter für Vergleich mit Green's Function
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
dof_coords = V.tabulate_dof_coordinates()
uh_values = uh.x.array
uh_interp = griddata(dof_coords, uh_values, (X, Y), method='cubic')
uh_interp = np.nan_to_num(uh_interp)

# Fehlerberechnung
exact_solution = greens_function_2d(X, Y, sources, strengths)
error = exact_solution - uh_interp
L2_error = np.sqrt(np.sum(error**2))
max_error = np.max(np.abs(error))

print(f"L2 Error: {L2_error:.2e}")
print(f"Max Error: {max_error:.2e}")

# Visualisierung
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Exakte Lösung
c1 = ax[0].contourf(X, Y, exact_solution, 50, cmap='viridis')
fig.colorbar(c1, ax=ax[0])
ax[0].set_title("Exakte Lösung (Green's Function)")

# Numerische Lösung
c2 = ax[1].contourf(X, Y, uh_interp, 50, cmap='viridis')
fig.colorbar(c2, ax=ax[1])
ax[1].set_title("Interpolierte numerische Lösung")

plt.show()
