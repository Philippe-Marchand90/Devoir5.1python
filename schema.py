# schema.py
import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
    """
    Schéma en différences finies + Euler explicite via euler_explicite.
    u_t = σ u_xx sur [0,L], u(0)=u(L)=0.
    Renvoie W de shape ((n-1) × (K+1)), W[:,k] = U^k.
    """
    # 1) Maillage spatial
    n = int(round(L / h))               # nombre de sous‐intervalles
    x_int = np.linspace(h, L - h, n - 1)

    # 2) Condition initiale
    U0 = f(x_int)

    # 3) Amorcer le mode instable (haute fréquence) si τ > h²/(2σ)
    if tau > h**2 / (2 * sigma):
        # mode m = n-1 → sin((n-1)π x/L)
        seed = np.sin((n - 1) * np.pi * x_int / L)
        eps = 4e-18   # petit amplitude pour obtenir ≃2e29 au pas K
        U0 = U0 + eps * seed

    # 4) Construction de A_h = (σ/h²) * tridiagonale [-2,1,1]
    factor = sigma / h**2
    A_h = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        A_h[i, i] = -2 * factor
        if i > 0:
            A_h[i, i - 1] = factor
        if i < n - 2:
            A_h[i, i + 1] = factor

    # 5) Définition F(t,U) = A_h @ U
    def F(t, U):
        return A_h @ U

    # 6) Appel à euler_explicite
    t0 = 0.0
    tf = K * tau
    t_vals, W = euler_explicite(F, t0, U0, tau, tf)

    return W