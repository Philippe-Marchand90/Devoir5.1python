import numpy as np

def construire_systeme(n, sigma, L):
    """
    Construit la fonction F(t, y) définissant le système d'équations différentielles
    issu de la discrétisation spatiale de l'équation de la chaleur.
    
    Paramètres :
        n     : Nombre de subdivisions de l'intervalle [0, L]
        sigma : Diffusivité thermique
        L     : Longueur de la barre
    
    Retour :
        F : fonction du temps t et du vecteur y (température), représentant le système d'EDO
    """
    h = L / n

    def F(t, y):
        dydt = np.zeros_like(y)
        dydt[0] = sigma / h**2 * (y[1] - 2*y[0])  # bord gauche
        for j in range(1, len(y)-1):
            dydt[j] = sigma / h**2 * (y[j+1] - 2*y[j] + y[j-1])
        dydt[-1] = sigma / h**2 * (-2*y[-1] + y[-2])  # bord droit
        return dydt

    return F