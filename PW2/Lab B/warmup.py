import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: Easy Convex Function ----------
def f(x):   return (x - 3)**2 + 1
def df(x):  return 2 * (x - 3)
def d2f(x): return 2.0

# 1. Gradient Descent by hand
x_gd = 0.0
lr = 0.1
for _ in range(100):
    x_gd = x_gd - lr * df(x_gd)
print("2A GD result:", x_gd)

# 2. Newton's method
x_newt = newton(df, 0, fprime=d2f)
print("2A Newton result:", x_newt)

# 3. SLSQP
res_slsqp = minimize(f, 0, method="SLSQP")
print("2A SLSQP result:", res_slsqp.x[0])

# ---------- 2B: Harder Landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

for x0 in [0, 2]:
    print(f"\n--- Starting from x0 = {x0} ---")
    # GD
    x_gd = float(x0)
    lr = 0.01
    for _ in range(1000):
        x_gd = x_gd - lr * dg(x_gd)
    print("2B GD:", x_gd)
    
    # Newton
    ans_newton = newton(dg, x0, fprime=d2g)
    print(f"2B Newton: {ans_newton:.4f} (d2g={d2g(ans_newton):.4f})")
    
    # SLSQP
    res_slsqp = minimize(g, x0, method="SLSQP")
    print("2B SLSQP:", res_slsqp.x[0])