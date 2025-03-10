import numpy as np
import matplotlib.pyplot as plt

# Load solutions
U_fem = np.load("fem_solution.npy")
U_analytical = np.load("analytical_solution.npy")

def compute_error(U_fem, U_analytical):
    """
    Compute absolute error, relative error, MSE, and max error between FEM and analytical solutions.
    """
    error_abs = np.abs(U_fem - U_analytical)
    error_rel = np.abs(U_fem - U_analytical) / (np.abs(U_analytical) + 1e-10)  # Avoid division by zero
    mse = np.mean(error_abs**2)  # Mean Squared Error
    max_error = np.max(error_abs)  # Maximum Error

    return error_abs, error_rel, mse, max_error
# Compute the error
error_abs, error_rel, mse, max_error = compute_error(U_fem, U_analytical)



def compute_l_2_error(U_fem, U_analytical):
    """
    Compute the L2 error between FEM and analytical solutions.
    """
    error = U_fem - U_analytical
    l_2_error = np.sqrt(np.sum(error**2))

    return error, l_2_error

error, l_2_error = compute_l_2_error(U_fem, U_analytical)


print(f"Mean Squared Error (MSE): {mse:.5e}")
print(f"Maximum Error: {max_error:.5e}")

print(f"L2 error: {l_2_error:.5e}")

# Plot Absolute Error
plt.imshow(error_abs, extent=[0, 1, 0, 1], origin='lower', cmap='inferno')
plt.colorbar(label='Absolute Error')
plt.title("Absolute Error between FEM and Analytical Solution")
plt.show()
