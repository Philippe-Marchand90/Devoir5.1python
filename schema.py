import numpy as np
from euler_explicite import euler_explicite

def schema(sigma, L, f, h, tau, K):
  
    n = int(round(L / h))

    factor = sigma / h**2
    A_h = np.zeros((n-1, n-1))
    for i in range(n-1):
        A_h[i, i] = -2 * factor
        if i > 0:
            A_h[i, i-1] = factor
        if i < n-2:
            A_h[i, i+1] = factor

    def F(t, U):
        return A_h @ U

    x_int = np.linspace(h, L-h, n-1)
    U0 = f(x_int)

    tf = K * tau
    t_vals, W = euler_explicite(F, t0=0.0, y0=U0, h=tau, tf=tf)

    return W