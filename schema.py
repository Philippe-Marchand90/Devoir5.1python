import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
    """
    Schéma par différences finies + Euler explicite
    via euler_explicite, mais en forçant l'étape
    directe « à la main » pour garder exactement
    la même erreur machine que votre version stable.

    Renvoie W shape = ((n-1) × (K+1)), W[:,k] = U^k
    """

    # maillage spatial
    n = int(round(L / h))
    x_int = np.linspace(h, L - h, n-1)

    # condition initiale
    U0 = f(x_int)

    # coefficient numérique
    mu = sigma * tau / h**2

    # on définit F(t,U) telle que Euler explicite
    # réalise exactement U_new = U + mu*(U_{j+1}-2U_j+U_{j-1})
    def F(t, U):
        U_new = np.empty_like(U)
        # bord gauche j=1
        U_new[0] = U[0] + mu*(U[1] - 2*U[0] + 0)
        # points intérieurs
        for j in range(1, n-2):
            U_new[j] = U[j] + mu*(U[j+1] - 2*U[j] + U[j-1])
        # bord droit j=n-1
        U_new[-1] = U[-1] + mu*(0 - 2*U[-1] + U[-2])
        # on renvoie le taux de variation d'U pour que
        # y_{k+1} = y_k + tau * F = U_new
        return (U_new - U) / tau

    # appel à euler_explicite : on obtient t_vals et W
    t0 = 0.0
    tf = K * tau
    t_vals, W = euler_explicite(F, t0, U0, tau, tf)

    return W