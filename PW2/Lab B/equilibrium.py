import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: Imbalance function
def k_imbalance(x):
    return ((2*x)**2) / ((a - x) * (b - x)) - K

# TODO 2: Newton's method
x_newton = newton(k_imbalance, x0=0.5)
print("Newton x:", x_newton)

# TODO 3: SLSQP
def objective(x):
    val = x[0] if isinstance(x, (list, np.ndarray)) else x
    return k_imbalance(val)**2

res = minimize(objective, x0=[0.5], bounds=[(0, 0.999)], method="SLSQP")
x_slsqp = res.x[0]
print("SLSQP x:", x_slsqp)

# TODO 4: Report and Plot
print(f"Equilibrium amounts: H2 = {a - x_newton:.3f} mol, I2 = {b - x_newton:.3f} mol, HI = {2*x_newton:.3f} mol")

x_vals = np.linspace(0, 0.99, 100)
plt.plot(x_vals, a - x_vals, label="H2 / I2")
plt.plot(x_vals, 2 * x_vals, label="HI")
plt.axvline(x_newton, color='red', linestyle='--', label=f"Equilibrium (x={x_newton:.2f})")
plt.xlabel("Extent x")
plt.ylabel("Amount (mol)")
plt.legend()
plt.savefig('equilibrium.png')
print("equilibrium.png saved successfully!")