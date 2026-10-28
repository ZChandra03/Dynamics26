import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a, b, c = 0.2, 0.2, 5.7

def f(q):
    x, y, z = q
    return np.array([-y-z, x+a*y, b+z*(x-c)])

# (a): integrate the driver and discard t < 100.
driver = solve_ivp(lambda t, q: f(q), (0, 300), [1, 1, 1],
                   t_eval=np.linspace(0, 300, 30001),
                   rtol=1e-9, atol=1e-11)
q = driver.y[:, driver.t >= 100]
ax = plt.figure().add_subplot(projection="3d")
ax.plot(*q, lw=0.5)
ax.set(xlabel="$x_D$", ylabel="$y_D$", zlabel="$z_D$")
plt.savefig("rossler_attractor.png", dpi=200, bbox_inches="tight")

# (d): the chosen controls equal f(D) - f(R) - (R-D).
def coupled(t, q):
    D, R = q[:3], q[3:]
    u = f(D) - f(R) - (R-D)
    return np.r_[f(D), f(R)+u]

t = np.linspace(0, 30, 3001)
sol = solve_ivp(coupled, (0, 30), [1, 1, 1, 2, -1, 3],
                t_eval=t, rtol=1e-9, atol=1e-11)
e = sol.y[3:] - sol.y[:3]
error = np.linalg.norm(e, axis=0)
print("Error norm at t=20:", error[2000])

plt.figure()
plt.semilogy(t, np.maximum(error, 1e-15), label="Numerical error norm")
plt.semilogy(t, 3*np.exp(-t), "--", label="$3e^{-t}$")
plt.xlabel("t")
plt.ylabel("Error norm")
plt.legend()
plt.tight_layout()
plt.savefig("rossler_sync.png", dpi=200)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for i, name in enumerate(["x", "y", "z"]):
    axes[i].plot(t, sol.y[i], label="Driver")
    axes[i].plot(t, sol.y[i+3], "--", label="Response")
    axes[i].set_ylabel(f"${name}$")

axes[0].legend()
axes[-1].set_xlabel("t")
fig.tight_layout()
fig.savefig("rossler_timeseries.png", dpi=200)

plt.show()