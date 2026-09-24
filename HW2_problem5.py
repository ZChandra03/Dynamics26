import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5
N = 50

x = np.linspace(-np.pi, np.pi, 5000)
W = np.zeros_like(x)

for n in range(N + 1):
    W += 2**(-n * alpha) * np.cos((2**n) * x)

plt.plot(x, W)
plt.xlabel("x")
plt.ylabel("W(x)")
plt.title(rf"$W(x)=\sum_{{n=0}}^{{{N}}}2^{{-n\alpha}}\cos(2^n x),\ \alpha={alpha}$")
plt.grid()
plt.show()