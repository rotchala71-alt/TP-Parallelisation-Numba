"""
TP Numba — Benchmark complémentaire (étape 4 détaillée).

Mesure le temps de la version parallèle pour 1, 2, 4, 8, 16 threads
(selon ce que la machine permet), calcule le speedup à chaque point,
déduit p par Amdahl, et trace la courbe de speedup (speedup.png).

Le speedup de référence T(1) est mesuré avec UN thread, ce qui est la
référence correcte pour évaluer le passage à l'échelle de prange.
"""
import sys
import json
import time
import numpy as np
from numba import set_num_threads, config

from moyenne import moyennes_parallele
from calcul_moyenne import charger_csv, meilleur_temps, amdahl_proportion


def main(fichier="etudiants.csv"):
    maths, physique, anglais = charger_csv(fichier)
    n = maths.shape[0]
    max_threads = config.NUMBA_NUM_THREADS

    # Liste des nombres de threads à tester, bornée par la machine
    candidats = [1, 2, 4, 8, 16, 32]
    threads_list = [t for t in candidats if t <= max_threads]
    if not threads_list:
        threads_list = [1]

    print(f"N = {n} | threads max = {max_threads} | tests = {threads_list}\n")

    # Warm-up
    set_num_threads(threads_list[0])
    _ = moyennes_parallele(maths[:100], physique[:100], anglais[:100])

    lignes = []
    t1 = None
    print(f"{'Threads':>8} | {'Temps (ms)':>12} | {'Speedup':>8} | {'p (Amdahl)':>10}")
    print("-" * 48)
    for th in threads_list:
        set_num_threads(th)
        t, _ = meilleur_temps(moyennes_parallele, (maths, physique, anglais))
        if t1 is None:
            t1 = t
        s = t1 / t
        p = amdahl_proportion(s, th)
        p_txt = f"{p:.4f}" if p is not None else "—"
        print(f"{th:>8} | {t * 1000:>12.4f} | {s:>8.3f} | {p_txt:>10}")
        lignes.append({"threads": th, "temps_ms": round(t * 1000, 4),
                       "speedup": round(s, 4),
                       "p": (round(p, 4) if p is not None else None)})

    # Sauvegarde
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"n_etudiants": n, "points": lignes}, f, indent=2, ensure_ascii=False)
    print("\nDonnées du benchmark -> benchmark.json")

    # Graphe
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        th = [l["threads"] for l in lignes]
        sp = [l["speedup"] for l in lignes]
        plt.figure(figsize=(7, 4.5))
        plt.plot(th, sp, "o-", color="#1a3a5c", linewidth=2, label="Speedup mesuré")
        plt.plot(th, th, "--", color="gray", label="Idéal (linéaire)")
        plt.xlabel("Nombre de threads (P)")
        plt.ylabel("Speedup  S = T(1) / T(P)")
        plt.title(f"Speedup — calcul de moyenne Numba (N={n:,})".replace(",", " "))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("speedup.png", dpi=130)
        print("Graphe -> speedup.png")
    except Exception as e:
        print("Graphe non généré :", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "etudiants.csv")
