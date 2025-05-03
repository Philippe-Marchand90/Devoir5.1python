import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
    """
    Schéma en différences finies + Euler explicite via euler_explicite.
    
    Résout u_t = σ u_xx sur [0,L], u(0)=u(L)=0, 
    renvoie W de shape ((n-1) × (K+1)), W[:,k] = U^k (points intérieurs).
    """
    # nombre de sous‐intervalles spatiaux
    n = int(round(L / h))
    
    # points intérieurs x_j = j*h, j=1…n-1
    x_int = np.linspace(h, L - h, n - 1)
    U0 = f(x_int)  # condition initiale
    
    # construction de la matrice tridiagonale A_h
    factor = sigma / h**2
    A_h = np.zeros((n - 1, n - 1))
    # diagonale principale
    np.fill_diagonal(A_h, -2 * factor)
    # sous- et sur-diagonales
    np.fill_diagonal(A_h[1:], factor)
    np.fill_diagonal(A_h[:,1:], factor)
    
    # dérivée temporelle F(t,U) = A_h @ U
    def F(t, U):
        return A_h @ U
    
    # appel à euler_explicite avec pas de temps tau jusqu'à tf = K*tau
    t0 = 0.0
    tf = K * tau
    t_vals, W = euler_explicite(F, t0, U0, tau, tf)
    
    return W
