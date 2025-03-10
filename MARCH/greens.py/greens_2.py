import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def poisson_solver_dirichlet(N, x_source, y_source, q=10):
    """
    Löst die Poisson-Gleichung mit Dirichlet-Randbedingungen auf einem N x N Gitter.
    """

    L = 1.0  # Domänengröße
    h = L / (N + 1)  # Gitterabstand
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    # Diskretisierte Laplace-Matrix mit Dirichlet-Randbedingungen
    main_diag = -4 * np.ones(N * N)
    off_diag = np.ones(N * N)
    A = sp.diags([main_diag, off_diag, off_diag, off_diag, off_diag],
                 [0, -1, 1, -N, N], shape=(N * N, N * N)).tolil()

    # Randbedingungen setzen (Dirichlet: u = 0 an den Rändern)
    for i in range(N):
        for j in range(N):
            index = i * N + j
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                A[index, :] = 0
                A[index, index] = 1  # Setzt Randwerte auf Null (Dirichlet)

    A = A.tocsc()  # Konvertiert in Sparse-Format

    # Rechte Seite der Poisson-Gleichung (Quelle als diskretisierte Delta-Funktion)
    b = np.zeros(N * N)
    src_idx = (int(y_source * N) * N) + int(x_source * N)  # Index der Quelle
    b[src_idx] = q / h**2  # Punktquelle ins Gitter setzen

    # Lösung des Gleichungssystems Ax = b
    u = spla.spsolve(A, b)

    return X, Y, u.reshape((N, N))

# Parameter
grid_size = 100
x_src, y_src = 0.5, 0.5  # Quelle in der Mitte

# Berechnung der Lösung
X, Y, U_numeric = poisson_solver_dirichlet(grid_size, x_src, y_src)

# Visualisierung
plt.imshow(U_numeric, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Potential u(x, y)')
plt.title("Numerische Lösung der Poisson-Gleichung mit Dirichlet-Randbedingungen")
plt.show()
