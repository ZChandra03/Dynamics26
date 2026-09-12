import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

# System of ODEs
def system(t, z):
    x, y = z
    dxdt = -y + x * (1 - x**2 - y**2)
    dydt =  x + y * (1 - x**2 - y**2)
    return [dxdt, dydt]


# ------------------------------------------------------------
# Nullclines
# ------------------------------------------------------------

x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)

X, Y = np.meshgrid(x, y)

# x' = 0
F = -Y + X * (1 - X**2 - Y**2)

# y' = 0
G = X + Y * (1 - X**2 - Y**2)


fig, ax = plt.subplots(figsize=(8, 8))

# Plot nullclines
ax.contour(
    X, Y, F,
    levels=[0],
    colors=["green"],
    linewidths=2
)

ax.contour(
    X, Y, G,
    levels=[0],
    colors=["red"],
    linewidths=2
)


# ------------------------------------------------------------
# Phase-space trajectories
# ------------------------------------------------------------

initial_conditions = [
    (0.1, 0),
    (0.4, 0.2),
    (-0.5, 0.3),
    (0.8, -0.4),
    (1.5, 0),
    (-1.5, 0.5),
    (0.5, 1.5),
    (-1.2, -1.2)
]

t_span = (0, 20)
t_eval = np.linspace(0, 20, 2000)

for z0 in initial_conditions:

    sol = solve_ivp(
        system,
        t_span,
        z0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11
    )

    ax.plot(
        sol.y[0],
        sol.y[1],
        color="black",
        linewidth=1.2
    )

    # Mark initial condition
    ax.plot(
        z0[0],
        z0[1],
        "ko",
        markersize=4
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.set_aspect("equal")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

ax.set_title("Nullclines and Phase-Space Trajectories")

# Custom legend
legend_elements = [
    Line2D([0], [0], color="green", lw=2,
           label=r"$\dot{x}=0$ nullcline"),
    Line2D([0], [0], color="red", lw=2,
           label=r"$\dot{y}=0$ nullcline"),
    Line2D([0], [0], color="black", lw=1.2,
           label="Trajectories")
]

ax.legend(handles=legend_elements)

plt.grid(alpha=0.25)
plt.tight_layout()

# Optional: save for Overleaf
plt.savefig("problem6_phase_plane.png", dpi=300)

plt.show()