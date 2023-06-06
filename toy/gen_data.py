import numpy as np
import matplotlib.pyplot as plt

TRN_MIN = -5
TRN_MAX = 5
TEST_MIN = -6
TEST_MAX = 6

N_TRN = 1000
N_TEST = 500
def generate_data(num_points=N_TRN, x_min=TEST_MIN, x_max=TRN_MAX):
    #np.random.seed(42)  # Set a seed for reproducibility

    x_values = np.random.uniform(x_min, x_max, num_points)
    mu_values = np.sin(x_values)
    sigma_values = 0.5 * (1 + np.exp(-x_values)) ** -1
    y_values = np.random.normal(mu_values, sigma_values ** 2)

    return x_values, y_values



x_values, y_values = generate_data()

plt.scatter(x_values, y_values, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()