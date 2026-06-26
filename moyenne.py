"""
TP Numba — Étapes 2 & 3 : Calcul de la moyenne pondérée.

Barème (coefficients) :
    Maths     (M) : 5
    Physique  (P) : 4
    Anglais   (A) : 2
    Moyenne = (M*5 + P*4 + A*2) / 11

Chaque étudiant est INDÉPENDANT des autres : aucune dépendance entre
l'itération i et l'itération i-1. C'est ce qui rend la parallélisation
correcte (mêmes résultats) et efficace (passage à l'échelle).

Ce module fournit :
    - moyennes_sequentiel  : @njit            (étape 2)
    - moyennes_parallele   : @njit(parallel)  (étape 3, prange)
"""
import numpy as np
from numba import njit, prange

# Coefficients des matières
COEF_MATHS = 5.0
COEF_PHYSIQUE = 4.0
COEF_ANGLAIS = 2.0
COEF_TOTAL = COEF_MATHS + COEF_PHYSIQUE + COEF_ANGLAIS  # = 11.0


# ----------------------------------------------------------------------
# ÉTAPE 2 — Version SÉQUENTIELLE (un seul thread)
# ----------------------------------------------------------------------
@njit(cache=True)
def moyennes_sequentiel(maths, physique, anglais):
    n = maths.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in range(n):                       # boucle classique, séquentielle
        res[i] = (maths[i] * COEF_MATHS
                  + physique[i] * COEF_PHYSIQUE
                  + anglais[i] * COEF_ANGLAIS) / COEF_TOTAL
    return res


# ----------------------------------------------------------------------
# ÉTAPE 3 — Version PARALLÈLE (multithread via prange)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def moyennes_parallele(maths, physique, anglais):
    n = maths.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):                      # prange => Numba répartit sur les cœurs
        res[i] = (maths[i] * COEF_MATHS
                  + physique[i] * COEF_PHYSIQUE
                  + anglais[i] * COEF_ANGLAIS) / COEF_TOTAL
    return res
