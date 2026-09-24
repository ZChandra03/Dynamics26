import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, np.pi, 500)

x1 = np.zeros_like(t)
x2 = 0.25 * np.sin(t)**2

plt.plot(t, x1, label=r"$x_1(t)=0$")
plt.plot(t, x2, label=r"$x_2(t)=\frac{1}{4}\sin^2(t)$")

plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()