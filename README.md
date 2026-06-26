# TP — Paralléliser un calcul de moyenne avec Numba

Calcul de la moyenne pondérée des notes de N ≥ 1 000 000 étudiants, en versions
**séquentielle** et **parallèle** (Numba `prange`), avec mesure du **speedup** et
déduction de la **proportion parallélisable** par la **loi d'Amdahl**.

> Moyenne = (Maths×5 + Physique×4 + Anglais×2) / 11

## Correspondance instructions → fichiers

| # | Instruction | Fichier |
|---|-------------|---------|
| 1 | Générer ≥ 1 000 000 d'étudiants en CSV | `generer_donnees.py` |
| 2 | Version séquentielle | `moyenne.py` → `moyennes_sequentiel` |
| 3 | Version parallèle | `moyenne.py` → `moyennes_parallele` |
| 4 | Calculer le speedup | `calcul_moyenne.py`, `benchmark.py` |
| 5 | Proportion parallélisable (Amdahl) | `calcul_moyenne.py` |
| 6 | Rapport détaillé + lien dépôt | `RAPPORT.md` / `RAPPORT.pdf`, `GUIDE_GIT.md` |

## Installation

```bash
pip install -r requirements.txt
```

## Exécution (chaîne complète)

```bash
python3 generer_donnees.py 1000000   # 1) crée etudiants.csv
python3 calcul_moyenne.py            # 2,3) calcule  4,5) speedup + Amdahl -> resultats.json
python3 benchmark.py                 # 4) courbe de speedup multi-threads -> speedup.png
python3 generer_rapport.py           # 6) régénère RAPPORT.md avec VOS mesures réelles
```

Le rapport se remplit automatiquement avec les mesures de votre machine.
Sur une machine multicœur, la proportion d'Amdahl est calculée et injectée.

## Régénérer le PDF du rapport

```bash
pip install markdown
python3 -c "import markdown,pathlib; \
html=markdown.markdown(pathlib.Path('RAPPORT.md').read_text(),extensions=['tables','fenced_code']); \
pathlib.Path('RAPPORT.html').write_text('<meta charset=utf-8>'+html)"
wkhtmltopdf RAPPORT.html RAPPORT.pdf   # ou : ouvrez RAPPORT.html et imprimez en PDF
```

## Idée clé

Aucune dépendance entre étudiants → problème *embarrassingly parallel*. La
parallélisation consiste à remplacer `range` par `prange` dans
`@njit(parallel=True)`. Numba compile au 1er appel : on fait un *warm-up* hors
chronométrage et on garde le meilleur temps sur 7 répétitions.
