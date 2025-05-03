import numpy as np


def schema(sigma, L, f, h, tau, K):
    """
    Schéma en différences finies + Euler explicite « à la main »
    pour u_t = σ u_xx sur [0,L], u(0)=u(L)=0.

    Renvoie W de forme (n-1, K+1) avec W[:,k] = u^k aux points intérieurs.
    """
    # nombre de sous-intervalles spatiaux
    n = int(round(L / h))

    # points intérieurs x_j = j*h, j=1…n-1
    x_int = np.linspace(h, L-h, n-1)
    U = f(x_int)               # U^0
    W = np.zeros((n-1, K+1))
    W[:, 0] = U.copy()

    # coefficient de diffusion numérique
    mu = sigma * tau / h**2

    for k in range(K):
        U_new = np.zeros_like(U)

        # j = 1 (U[0]) avec u(0)=0
        U_new[0] = U[0] + mu*(U[1] - 2*U[0] + 0)

        # j = 2…n-2 (U[1]…U[n-3])
        for j in range(1, n-2):
            U_new[j] = U[j] + mu*(U[j+1] - 2*U[j] + U[j-1])

        # j = n-1 (U[n-2]) avec u(L)=0
        U_new[-1] = U[-1] + mu*(0 - 2*U[-1] + U[-2])

        # sauvegarde et avance
        W[:, k+1] = U_new
        U = U_new

    return W