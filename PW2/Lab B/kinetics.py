import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: Read kinetics.csv
data = np.loadtxt('kinetics.csv', delimiter=',', skiprows=1)
t, C = data[:, 0], data[:, 1]
C0 = C[0]

# TODO 2: Total error function
def total_error(k):
    k_val = k[0] if isinstance(k, (list, np.ndarray)) else k
    return np.sum((C - C0 * np.exp(-k_val * t))**2)

# TODO 3: Minimise total error
res = minimize(total_error, x0=[0.5], bounds=[(0, 5)], method="SLSQP")
fitted_k = res.x[0]
print("Fitted k:", fitted_k)

# TODO 4: Plot
plt.scatter(t, C, color='black', label="Measured Data")
plt.plot(t, C0 * np.exp(-fitted_k * t), color='red', label=f"Fitted Curve (k={fitted_k:.3f})")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.savefig('kinetics.png')
print("kinetics.png saved successfully!")