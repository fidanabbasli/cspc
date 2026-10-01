import numpy as np
import matplotlib.pyplot as plt

# TODO 1: Read titration.csv
data = np.loadtxt('titration.csv', delimiter=',', skiprows=1)
V, pH = data[:, 0], data[:, 1]

# TODO 2: Compute slope and find peak
slope = np.gradient(pH, V)
eq_idx = np.argmax(slope)
eq_vol = V[eq_idx]
print("Equivalence Volume:", eq_vol, "mL")

# TODO 3: Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(V, pH, color='blue')
ax1.axvline(eq_vol, color='red', linestyle='--', label=f'Eq Point ({eq_vol} mL)')
ax1.set_xlabel('Volume Base (mL)')
ax1.set_ylabel('pH')
ax1.set_title('Titration Curve')
ax1.legend()

ax2.plot(V, slope, color='green')
ax2.axvline(eq_vol, color='red', linestyle='--', label=f'Peak ({eq_vol} mL)')
ax2.set_xlabel('Volume Base (mL)')
ax2.set_ylabel('d(pH)/dV')
ax2.set_title('Slope of Curve')
ax2.legend()

plt.tight_layout()
plt.savefig('titration.png')
print("titration.png saved successfully!")