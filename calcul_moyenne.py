"""
TP Numba — Étapes 4 & 5 : Speedup et loi d'Amdahl.

- Charge le CSV.
- Mesure le temps SÉQUENTIEL et le temps PARALLÈLE (meilleur sur N répétitions).
- Calcule le speedup S = T_seq / T_par.
- Déduit la proportion parallélisable p via la loi d'Amdahl.
- Vérifie que les deux versions donnent le MÊME résultat.
- Sauvegarde les mesures dans 'resultats.json' (utilisé pour le rapport).

Méthodologie :
  Numba compile au 1er appel -> on fait un WARM-UP hors chronométrage,
  puis on garde le MEILLEUR temps sur plusieurs répétitions.
"""
import sys
import json
import time
import numpy as np

from numba import get_num_threads, config
from moyenne import (
    moyennes_sequentiel,
    moyennes_parallele,
    COEF_MATHS, COEF_PHYSIQUE, COEF_ANGLAIS, COEF_TOTAL,
)


def charger_csv(fichier):
    """Charge le CSV en 3 tableaux numpy float64 (maths, physique, anglais)."""
    data = np.loadtxt(fichier, delimiter=",", skiprows=1)
    maths = np.ascontiguousarray(data[:, 1])
    physique = np.ascontiguousarray(data[:, 2])
    anglais = np.ascontiguousarray(data[:, 3])
    return maths, physique, anglais


def meilleur_temps(fonction, args, repetitions=7):
    """Retourne le meilleur temps (s) et le résultat de la fonction."""
    res = None
    best = float("inf")
    for _ in range(repetitions):
        t0 = time.perf_counter()
        res = fonction(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, res


def amdahl_proportion(speedup, p_proc):
    """
    Loi d'Amdahl : S = 1 / ((1-p) + p/P)
    Inversion    : p = P*(S-1) / (S*(P-1))
    Retourne la proportion parallélisable p (None si P<=1).
    """
    if p_proc <= 1:
        return None
    return p_proc * (speedup - 1) / (speedup * (p_proc - 1))


def main(fichier="etudiants.csv"):
    print("=" * 60)
    print("TP NUMBA — Calcul de moyenne séquentiel vs parallèle")
    print("=" * 60)

    print(f"\n[1] Chargement de '{fichier}'...")
    maths, physique, anglais = charger_csv(fichier)
    n = maths.shape[0]
    n_threads = get_num_threads()
    print(f"    N = {n} étudiants")
    print(f"    Coefficients : M={COEF_MATHS}, P={COEF_PHYSIQUE}, "
          f"A={COEF_ANGLAIS}, total={COEF_TOTAL}")
    print(f"    Threads Numba disponibles (P) : {n_threads}")
    print(f"    Cœurs détectés par le système : {config.NUMBA_NUM_THREADS}")

    # --- Warm-up : compilation JIT hors mesure ---
    print("\n[2] Compilation JIT (warm-up, hors chronométrage)...")
    _ = moyennes_sequentiel(maths[:100], physique[:100], anglais[:100])
    _ = moyennes_parallele(maths[:100], physique[:100], anglais[:100])

    # --- Mesures ---
    print("\n[3] Mesure des temps (meilleur sur 7 répétitions)...")
    t_seq, r_seq = meilleur_temps(moyennes_sequentiel, (maths, physique, anglais))
    t_par, r_par = meilleur_temps(moyennes_parallele, (maths, physique, anglais))

    # --- Vérification d'exactitude ---
    correct = bool(np.allclose(r_seq, r_par))
    print(f"    Résultats identiques (seq == par) : {correct}")
    if not correct:
        raise RuntimeError("Les versions séquentielle et parallèle diffèrent !")

    # --- Speedup (étape 4) ---
    speedup = t_seq / t_par

    # --- Amdahl (étape 5) ---
    p = amdahl_proportion(speedup, n_threads)

    print("\n" + "-" * 60)
    print(f"  Temps séquentiel T_seq : {t_seq * 1000:10.4f} ms")
    print(f"  Temps parallèle  T_par : {t_par * 1000:10.4f} ms")
    print(f"  SPEEDUP  S = T_seq/T_par = {speedup:8.4f}")
    print(f"  Processeurs P           = {n_threads}")
    if p is not None:
        smax = 1.0 / (1.0 - p) if p < 1 else float("inf")
        print(f"  PROPORTION PARALLÉLISABLE  p = {p:.4f}  ({p * 100:.2f} %)")
        print(f"  Speedup max théorique (P→∞) = 1/(1-p) = {smax:.2f}")
    else:
        print("  PROPORTION PARALLÉLISABLE : non calculable (P = 1 seul cœur).")
        print("  -> Exécutez ce script sur une machine MULTICŒUR pour Amdahl.")
    print("-" * 60)

    moyenne_classe = float(r_seq.mean())
    print(f"\n  Moyenne générale de la classe : {moyenne_classe:.4f} / 20")

    # --- Sauvegarde des mesures pour le rapport ---
    resultats = {
        "n_etudiants": int(n),
        "n_threads": int(n_threads),
        "t_seq_ms": round(t_seq * 1000, 4),
        "t_par_ms": round(t_par * 1000, 4),
        "speedup": round(speedup, 4),
        "p_parallelisable": (round(p, 4) if p is not None else None),
        "speedup_max_theorique": (round(1 / (1 - p), 2) if p is not None and p < 1 else None),
        "resultats_identiques": correct,
        "moyenne_classe": round(moyenne_classe, 4),
        "coefficients": {"maths": COEF_MATHS, "physique": COEF_PHYSIQUE,
                         "anglais": COEF_ANGLAIS, "total": COEF_TOTAL},
    }
    with open("resultats.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print("\n  Mesures sauvegardées dans 'resultats.json'")

    return resultats


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else "etudiants.csv"
    main(fichier)
