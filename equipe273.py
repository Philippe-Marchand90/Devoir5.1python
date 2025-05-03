import numpy as np
import matplotlib.pyplot as plt
from schema import schema

def main():
    sigma = 1.0
    L     = 1.0
    h     = 1/10
    f     = lambda x: np.sin(np.pi * x)

    # on trace pour τ=0.01 puis τ=0.001
    for tau in [0.01, 0.001]:
        K = int(1.0 / tau)           # pour atteindre t=1
        W = schema(sigma, L, f, h, tau, K)

        # construction du vecteur x de 0 à L (incluant les extrémités)
        n = int(round(L / h))
        x = np.linspace(0, L, n+1)

        # indices correspondant à t = 0, 0.05, 0.1 et 1
        t_indices = [
            0,
            int(0.05 / tau),
            int(0.1  / tau),
            int(1.0  / tau)
        ]

        plt.figure()

        # approximation numérique
        for idx in t_indices:
            t = idx * tau
            u = np.zeros(n+1)
            u[1:-1] = W[:, idx]
            plt.plot(x, u, label=f"approx et t={t:g}")

        # solution exacte u(x,t) = e^{-π²t} sin(πx)
        for idx in t_indices:
            t = idx * tau
            u_ex = np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
            plt.plot(x, u_ex, label=f"exa et t={t:g}")

        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Température pour tau={tau}")
        plt.legend()

    # affiche les deux figures
    plt.show()

if __name__ == "__main__":
    main()