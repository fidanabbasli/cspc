import time
from decay import simulate, simulate_loop

N0 = 50000
lam = 0.4

t0 = time.perf_counter()
simulate_loop(N0, lam)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
simulate(N0, lam)
t_vec = time.perf_counter() - t0

print(f"Loop execution time:       {t_loop:.4f} second")
print(f"NumPy execution time:      {t_vec:.4f} second")
print(f"NumPy {t_loop / t_vec:.1f} times faster!")