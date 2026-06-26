"""
TP Numba — Étape 6 : Génération du rapport détaillé.

Lit 'resultats.json' et 'benchmark.json' (produits par les scripts de mesure)
et génère RAPPORT.md en y injectant les VRAIES mesures de la machine.

Si la machine n'a qu'un seul cœur (P=1, Amdahl non calculable), le rapport
inclut clairement cette limitation ET un exemple de référence sur 4 cœurs
entièrement calculé, afin que l'étape 5 (Amdahl) reste illustrée.

Usage : python3 generer_rapport.py
"""
import json
import os

GITHUB_URL = "https://github.com/<votre-compte>/tp-numba-moyenne"


def charger(fichier):
    if os.path.exists(fichier):
        with open(fichier, encoding="utf-8") as f:
            return json.load(f)
    return None


def bloc_mesures(res, bench):
    if res is None:
        return ("> Les mesures n'ont pas encore été produites. Lancez "
                "`python3 calcul_moyenne.py` puis `python3 generer_rapport.py`.\n")

    n = res["n_etudiants"]
    P = res["n_threads"]
    lignes = []
    lignes.append(f"- **Nombre d'étudiants (N)** : {n:,}".replace(",", " "))
    lignes.append(f"- **Processeurs / threads (P)** : {P}")
    lignes.append(f"- **Temps séquentiel T_seq** : {res['t_seq_ms']} ms")
    lignes.append(f"- **Temps parallèle T_par** : {res['t_par_ms']} ms")
    lignes.append(f"- **Speedup S = T_seq / T_par** : **{res['speedup']}**")
    lignes.append(f"- **Résultats séquentiel == parallèle** : {res['resultats_identiques']}")
    lignes.append(f"- **Moyenne générale de la classe** : {res['moyenne_classe']} / 20")
    txt = "\n".join(lignes) + "\n"

    # Tableau du benchmark multi-threads
    if bench and bench.get("points"):
        txt += "\n**Détail du passage à l'échelle (benchmark) :**\n\n"
        txt += "| Threads (P) | Temps (ms) | Speedup S | p (Amdahl) |\n"
        txt += "|:-----------:|:----------:|:---------:|:----------:|\n"
        for pt in bench["points"]:
            p = pt["p"] if pt["p"] is not None else "—"
            txt += f"| {pt['threads']} | {pt['temps_ms']} | {pt['speedup']} | {p} |\n"
        txt += "\n"
    return txt


def bloc_amdahl(res):
    P = res["n_threads"] if res else 1
    if res and res.get("p_parallelisable") is not None:
        p = res["p_parallelisable"]
        S = res["speedup"]
        smax = res["speedup_max_theorique"]
        return (
            f"À partir de la mesure réelle (**S = {S}** sur **P = {P}** cœurs), "
            f"on inverse la loi d'Amdahl :\n\n"
            f"```\n"
            f"p = P*(S-1) / (S*(P-1))\n"
            f"p = {P}*({S}-1) / ({S}*({P}-1))\n"
            f"p ≈ {p}   →   {p*100:.2f} % parallélisable\n"
            f"```\n\n"
            f"Speedup maximal théorique (P→∞) : `1/(1-p) ≈ {smax}`.\n"
        )
    # Cas mono-cœur : exemple de référence sur 4 cœurs
    S_ex, P_ex = 3.65, 4
    p_ex = P_ex * (S_ex - 1) / (S_ex * (P_ex - 1))
    smax_ex = 1 / (1 - p_ex)
    return (
        f"> ⚠️ **Limitation de l'environnement de mesure** : la machine utilisée "
        f"n'expose qu'**un seul cœur** (P = {P}). Le speedup parallèle ne peut "
        f"donc pas y être mesuré (S ≈ 1) — c'est une limite **matérielle**, pas "
        f"un défaut du code. Sur une machine multicœur, `calcul_moyenne.py` "
        f"calcule automatiquement p et met à jour ce rapport.\n\n"
        f"**Exemple de référence entièrement calculé (machine 4 cœurs)** :\n\n"
        f"En supposant un speedup mesuré **S = {S_ex}** sur **P = {P_ex}** cœurs :\n\n"
        f"```\n"
        f"p = P*(S-1) / (S*(P-1))\n"
        f"p = {P_ex}*({S_ex}-1) / ({S_ex}*({P_ex}-1))\n"
        f"p = {P_ex*(S_ex-1):.2f} / {S_ex*(P_ex-1):.2f}\n"
        f"p ≈ {p_ex:.4f}   →   {p_ex*100:.2f} % parallélisable\n"
        f"```\n\n"
        f"Speedup maximal théorique (P→∞) : `1/(1-p) ≈ {smax_ex:.1f}`.\n"
    )


def main():
    res = charger("resultats.json")
    bench = charger("benchmark.json")

    rapport = f"""# Rapport — Paralléliser un calcul de moyenne avec Numba

**TP :** Calcul parallèle · **Niveau :** Intermédiaire · **Modalité :** Individuel

---

## 1. Présentation

On calcule la **moyenne pondérée** des notes de **N ≥ 1 000 000 étudiants**.

| Matière | Coefficient |
|:--------|:-----------:|
| Maths (M) | 5 |
| Physique (P) | 4 |
| Anglais (A) | 2 |

> **Moyenne = (M × 5 + P × 4 + A × 2) / 11**

Chaque étudiant est **indépendant** des autres : la moyenne de l'étudiant *i*
ne dépend pas de celle de l'étudiant *i−1*. Il n'existe **aucune dépendance de
données** entre les itérations. C'est un problème *embarrassingly parallel* :
la parallélisation est à la fois **correcte** (mêmes résultats que le séquentiel)
et **efficace** (passage à l'échelle proche du linéaire).

---

## 2. Génération des données (instruction 1)

`generer_donnees.py` crée un CSV de N étudiants (`id, maths, physique, anglais`),
notes tirées uniformément dans [0, 20], graine fixe (`seed=42`) pour la
reproductibilité.

```python
rng = np.random.default_rng(42)
maths    = rng.uniform(0, 20, n)
physique = rng.uniform(0, 20, n)
anglais  = rng.uniform(0, 20, n)
```

```
id,maths,physique,anglais
0,15.48,13.98,16.52
1,8.78,9.23,19.49
...
```

---

## 3. Version séquentielle (instruction 2)

Décorateur `@njit` : compilation en code machine, exécution sur **un seul thread**.

```python
@njit(cache=True)
def moyennes_sequentiel(maths, physique, anglais):
    n = maths.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        res[i] = (maths[i]*5 + physique[i]*4 + anglais[i]*2) / 11
    return res
```

---

## 4. Version parallèle (instruction 3)

`@njit(parallel=True)` + **`prange`** : Numba répartit automatiquement les
itérations sur les cœurs disponibles. **Une seule ligne change** (`range` →
`prange`).

```python
@njit(parallel=True, cache=True)
def moyennes_parallele(maths, physique, anglais):
    n = maths.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):                 # boucle parallèle
        res[i] = (maths[i]*5 + physique[i]*4 + anglais[i]*2) / 11
    return res
```

**Correction garantie** : chaque thread écrit `res[i]` pour des indices *i*
disjoints → aucune *race condition*. Le code vérifie `np.allclose(seq, par)`.

> **Méthodologie de mesure** : Numba compile au 1er appel. On fait un *warm-up*
> hors chronométrage, puis on garde le **meilleur temps sur 7 répétitions**.

---

## 5. Speedup mesuré (instruction 4)

{bloc_mesures(res, bench)}

![Courbe de speedup](speedup.png)

---

## 6. Loi d'Amdahl — proportion parallélisable (instruction 5)

La loi d'Amdahl relie le speedup *S*, le nombre de processeurs *P* et la
proportion parallélisable *p* :

> **S(P) = 1 / ( (1 − p) + p / P )**

En inversant :

> **p = P · (S − 1) / ( S · (P − 1) )**

{bloc_amdahl(res)}

**Interprétation** : une valeur de *p* proche de 1 confirme le caractère
massivement parallèle du problème — presque tout le travail tient dans la boucle
indépendante, et seule une faible fraction (mémoire, gestion des threads) reste
séquentielle. La loi d'Amdahl montre aussi que le speedup **plafonne** à
`1/(1−p)` quel que soit le nombre de cœurs.

---

## 7. Conclusion

- Problème **sans dépendance entre itérations** → parallélisation **correcte**
  et **naturelle** avec `prange`.
- Coût d'implémentation minimal : **une ligne** modifiée.
- Le **speedup** mesuré reste sous l'idéal linéaire (overhead mémoire + threads).
- La **loi d'Amdahl** quantifie la proportion parallélisable et le speedup
  maximal atteignable.

---

## 8. Reproduction

```bash
pip install -r requirements.txt
python3 generer_donnees.py 1000000     # étape 1
python3 calcul_moyenne.py              # étapes 2,3,4,5 + resultats.json
python3 benchmark.py                   # courbe de speedup
python3 generer_rapport.py             # régénère ce rapport avec vos mesures
```

---

## 9. Dépôt du code (instruction 6)

**Lien GitHub/GitLab :** {GITHUB_URL}

*(À remplacer par votre URL réelle après le `git push` — voir `GUIDE_GIT.md`.)*
"""

    with open("RAPPORT.md", "w", encoding="utf-8") as f:
        f.write(rapport)
    print("RAPPORT.md généré"
          + (" avec les mesures réelles." if res and res.get("p_parallelisable")
             else " (mono-cœur : exemple de référence inclus pour Amdahl)."))


if __name__ == "__main__":
    main()
