import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
    """
    Schéma en différences finies + Euler explicite
    pour u_t = σ u_xx sur [0,L] avec u(0)=u(L)=0.

    Renvoie W de forme (n-1, K+1) où chaque colonne k est U^k.
    """
    # nombre de sous-intervalles spatiaux
    n = int(round(L / h))

    # Construction de la matrice tridiagonale A_h
    factor = sigma / h**2
    A_h = np.zeros((n-1, n-1))
    for i in range(n-1):
        A_h[i, i] = -2 * factor
        if i > 0:
            A_h[i, i-1] = factor
        if i < n-2:
            A_h[i, i+1] = factor

    # Définition de F(t, U) = A_h @ U
    def F(t, U):
        return A_h @ U

    # Condition initiale U^0 = [f(h), f(2h), …, f((n-1)h)]
    x_int = np.linspace(h, L-h, n-1)
    U0 = f(x_int)

    # On intègre de t0=0 à tf = K*tau avec pas h = tau
    t0 = 0.0
    tf = K * tau
    t_vals, W = euler_explicite(F, t0=t0, y0=U0, h=tau, tf=tf)

    # W est de taille (n-1)×(K+1) : W[:, k] = U^k
    return W
