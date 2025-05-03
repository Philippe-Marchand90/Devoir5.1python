import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
    """
    Résout u_t = σ u_xx sur [0,L] avec u(0,t)=u(L,t)=0
    par différences finies en espace et Euler explicite en temps.

    Entrées
    -------
    sigma : diffusivité (float)
    L     : longueur de la barre (float)
    f     : fonction initiale f(x)
    h     : pas spatial
    tau   : pas temporel
    K     : nombre d’itérations (t_max = K*tau)

    Sortie
    ------
    W : tableau de forme ((n-1) × (K+1)), dont chaque colonne k est U^k
        contenant les valeurs u_j^k pour j=1…n−1
    """
    # nombre de sous-intervalles spatiaux
    n = int(round(L / h))

    # construction de la matrice A_h
    factor = sigma / h**2
    A_h = np.zeros((n-1, n-1))
    for i in range(n-1):
        A_h[i, i] = -2 * factor
        if i > 0:
            A_h[i, i-1] = factor
        if i < n-2:
            A_h[i, i+1] = factor

    # définition de F(t, U) = A_h @ U
    def F(t, U):
        return A_h @ U

    # condition initiale U^0 = (f(h), f(2h), …, f((n-1)h))
    x_int = np.linspace(h, L-h, n-1)
    U0 = f(x_int)

    # on résout de t=0 à t=K*tau
    tf = K * tau
    t_vals, W = euler_explicite(F, t0=0.0, y0=U0, h=tau, tf=tf)

    return W