"""
TP Numba — Étape 1 : Génération des données.

Génère un fichier CSV d'au moins 1 000 000 d'étudiants.
Chaque étudiant a 3 notes sur 20 : Maths (M), Physique (P), Anglais (A).

Colonnes du CSV : id, maths, physique, anglais
"""
import numpy as np
import sys
import time


def generer(n: int = 1_000_000, fichier: str = "etudiants.csv", seed: int = 42) -> None:
    """Génère n étudiants avec des notes uniformes dans [0, 20] et les écrit en CSV."""
    if n < 1_000_000:
        print(f"Attention : le TP demande au moins 1 000 000 d'étudiants (n={n}).")

    rng = np.random.default_rng(seed)
    ids = np.arange(n, dtype=np.int64)
    maths = rng.uniform(0, 20, n)
    physique = rng.uniform(0, 20, n)
    anglais = rng.uniform(0, 20, n)

    t0 = time.perf_counter()
    # Écriture vectorisée et rapide via np.savetxt
    data = np.column_stack([ids, maths, physique, anglais])
    np.savetxt(
        fichier,
        data,
        fmt=["%d", "%.2f", "%.2f", "%.2f"],
        delimiter=",",
        header="id,maths,physique,anglais",
        comments="",
    )
    dt = time.perf_counter() - t0
    print(f"{n} étudiants écrits dans '{fichier}' en {dt:.2f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    fichier = sys.argv[2] if len(sys.argv) > 2 else "etudiants.csv"
    generer(n, fichier)
