from visualisation import greens_function

# Prompt the user for the point to evaluate the potential
x_point, y_point = map(float, input('Enter the (x,y) coordinates of the point to evaluate the potential (e.g. 0.5,0.5): ').split(','))

# Define the charge location
x_charge, y_charge = 0.5, 0.5

# Calculate the potential at the specified point
spec_potential = greens_function(x_point, y_point, x_charge, y_charge)
print(f"The potential at point ({x_point}, {y_point}) is: {spec_potential}")
