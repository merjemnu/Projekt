import torch
import torch.nn as nn
import numpy as np

# Define the neural network (PINN)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.hidden_layers(x)

# Define the PDE residual loss (Poisson equation)
def pde_loss(model, interior_points):
    interior_points.requires_grad_(True)
    u = model(interior_points)

    # Compute second derivatives (Laplacian)
    grads = torch.autograd.grad(u, interior_points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    laplacian = torch.autograd.grad(grads, interior_points, grad_outputs=torch.ones_like(grads), create_graph=True)[0]
    
    # Poisson equation residual Δu = -6
    residual = laplacian.sum(dim=1) + 6
    return torch.mean(residual**2)

# Define boundary loss
def boundary_loss(model, boundary_points, boundary_values):
    u_pred = model(boundary_points)
    return torch.mean((u_pred - boundary_values) ** 2)

# Generate training data: Interior and boundary points
num_interior = 1000
num_boundary = 200

# Interior points in [0,1]x[0,1]
interior_points = torch.rand((num_interior, 2), requires_grad=True)

# Boundary points
boundary_x = torch.cat([torch.rand(num_boundary // 4, 1), torch.zeros(num_boundary // 4, 1)], dim=1)  # x-boundary
boundary_y = torch.cat([torch.zeros(num_boundary // 4, 1), torch.rand(num_boundary // 4, 1)], dim=1)  # y-boundary
boundary_points = torch.cat([boundary_x, boundary_y], dim=0)

# Compute boundary condition values: u(x,y) = 1 + x^2 + 2y^2
boundary_values = 1 + boundary_points[:, 0]**2 + 2 * boundary_points[:, 1]**2
boundary_values = boundary_values.view(-1, 1)

# Training
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = pde_loss(model, interior_points) + boundary_loss(model, boundary_points, boundary_values)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Predict solution on a grid
X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
u_pred = model(grid).detach().numpy().reshape(X.shape)

# Plot the solution
import matplotlib.pyplot as plt
plt.contourf(X, Y, u_pred, levels=50, cmap="viridis")
plt.colorbar(label="u(x,y)")
plt.title("PINN Solution for Poisson's Equation")
plt.show()
    