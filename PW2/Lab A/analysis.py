import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]


v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Mean acceleration: {a.mean():.2f} m/s^2")
print(f"Acceleration std dev: {a.std():.2f} m/s^2")


v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - y_rec))
print(f"Max difference (original vs recovered): {max_diff:.4f} m")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

ax1.plot(t, y, label='Position (y)', color='blue')
ax1.set_ylabel('Position (m)')
ax1.legend()

ax2.plot(t, v, label='Velocity (v)', color='orange')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()

ax3.plot(t, a, label='Acceleration (a)', color='green')
ax3.axhline(-9.81, color='red', linestyle='--', label='-9.81 m/s²')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.set_xlabel('Time (s)')
ax3.legend()

plt.tight_layout()
plt.savefig('motion.png')
plt.show()


traj_data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t_tr = traj_data[:, 0]
x_tr = traj_data[:, 1]
y_tr = traj_data[:, 2]

vx = np.gradient(x_tr, t_tr)
vy = np.gradient(y_tr, t_tr)
speed = np.sqrt(vx**2 + vy**2)

fig2, (ax_path, ax_speed) = plt.subplots(1, 2, figsize=(12, 5))

ax_path.plot(x_tr, y_tr, color='purple')
ax_path.set_title('Trajectory Path (x vs y)')
ax_path.set_xlabel('x (m)')
ax_path.set_ylabel('y (m)')

ax_speed.plot(t_tr, speed, color='teal')
ax_speed.set_title('Speed vs Time')
ax_speed.set_xlabel('Time (s)')
ax_speed.set_ylabel('Speed (m/s)')

plt.tight_layout()
plt.show()