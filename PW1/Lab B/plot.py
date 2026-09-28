import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

t, observed = np.loadtxt('decay_observed.csv', delimiter=',', skiprows=1, unpack=True)

N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

ax1.scatter(t, observed, color='blue', alpha=0.7)
ax1.set_title('Observed data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Count')
ax1.grid(True)

ax2.plot(t, analytical, color='red')
ax2.set_title('Analytical')
ax2.set_xlabel('Time')
ax2.grid(True)

plt.tight_layout()

plt.savefig('figure.png')
print("Picture saved!")