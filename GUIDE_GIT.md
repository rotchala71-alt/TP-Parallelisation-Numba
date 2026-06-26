# Mise en ligne du dépôt (instruction 6)

```bash
cd tp_numba
git init
git add generer_donnees.py moyenne.py calcul_moyenne.py benchmark.py \
        generer_rapport.py requirements.txt README.md RAPPORT.md \
        RAPPORT.pdf GUIDE_GIT.md .gitignore
git commit -m "TP Numba : moyenne sequentiel + parallele, speedup, Amdahl"

# GitHub : creez un depot vide "tp-numba-moyenne", puis :
git remote add origin https://github.com/<votre-compte>/tp-numba-moyenne.git
git branch -M main
git push -u origin main
```

Collez ensuite l'URL obtenue dans la section 9 de RAPPORT.md, puis
regenerez le PDF (voir README). Le CSV (~24 Mo) est ignore : il se
recree avec `python3 generer_donnees.py 1000000`.
