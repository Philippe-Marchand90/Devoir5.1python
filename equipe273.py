import numpy as np
import matplotlib.pyplot as plt
from euler_explicite import euler_explicite
from schema import construire_systeme

# Paramètres du problème
L = 1.0
sigma = 0.01
n = 20
h_espace = L / n
x_vals = np.linspace(h_espace, L - h_espace, n - 1)

# Condition initiale
f = lambda x: np.sin(np.pi * x)
y0 = f(x_vals)

# Discrétisation en temps
t0 = 0
tf = 1
tau = 0.001

# Construction du système et résolution
F = construire_systeme(n, sigma, L)
t_vals, y_vals = euler_explicite(F, t0, y0, tau, tf)

# Première figure : début, milieu, fin
plt.figure()
plt.plot(x_vals, y_vals[:, 0], label='t = 0')
plt.plot(x_vals, y_vals[:, len(t_vals)//2], label=f't = {t_vals[len(t_vals)//2]:.2f}')
plt.plot(x_vals, y_vals[:, -1], label=f't = {t_vals[-1]:.2f}')
plt.xlabel('x')
plt.ylabel('Température u(x,t)')
plt.title('Évolution de la température (3 temps)')
plt.legend()
plt.grid(True)
plt.show()

# Deuxième figure : plusieurs temps
plt.figure()
for i in range(0, len(t_vals), max(1, len(t_vals)//10)):
    plt.plot(x_vals, y_vals[:, i], label=f"t = {t_vals[i]:.2f}")
plt.xlabel('x')
plt.ylabel('Température u(x,t)')
plt.title('Température à différents temps')
plt.legend()
plt.grid(True)
plt.show()