# Petits VLMs : légender ou juger des images avec quantification

Projet 7. Comparaison de deux petits modèles vision-langage en fp16, 8 bits et 4 bits sur Flickr8k, dans la limite de 12 Go de VRAM (Google Colab / Kaggle).

**Question :** que perd-on en qualité quand on quantifie, et que gagne-t-on en mémoire et en vitesse ?

## Équipe
| Membre | Rôle / contributions |
|---|---|
| _Nom 1_ | |
| _Nom 2_ | |

## Option choisie
- [x] A : légendes pour entraîner une LoRA
- [ ] B : jugement du VLM comparé aux humains

## Installation (Google Colab, GPU T4)
```python
!git clone https://github.com/<utilisateur>/vlm-quantization-project.git
%cd vlm-quantization-project
!pip install -q -r requirements.txt
!python data/download_flickr8k.py --n 500
```

## Lancement
- Expériences : exécuter les notebooks de `notebooks/` dans l'ordre.
- Démo : `!python demo/app.py` (lien public Gradio affiché).

## Résultats principaux
_Tableau modèle × précision → qualité, VRAM, latence (à remplir)._

## Structure
| Dossier | Contenu |
|---|---|
| `notebooks/` | 01 données, 02 baseline, 03 modèles/quantification, 04 évaluation, 05 démo |
| `src/` | Code réutilisable (`models.py`, `metrics.py`) |
| `configs/` | Paramètres des expériences |
| `data/` | Script de téléchargement et fiche des données |
| `results/` | Figures, tableaux, exemples |
| `demo/` | Application Gradio et vidéo |
| `report/` | Rapport final (6 à 8 pages) |
| `docs/` | Fiche modèle, éthique, licences |

## Règles de l'équipe
Commits réguliers et précis, tâches via issues, pas de gros fichiers dans Git.

## Option A : périmètre retenu
Génération de légendes par 2 VLMs en fp16 / 8 bits / 4 bits sur les mêmes images, évaluées avec BLEU-4, CIDEr, ROUGE-L, METEOR (contre les 5 légendes Flickr8k), avec VRAM, latence et taille. Les légendes produites constituent le jeu destiné à entraîner une LoRA. Une mini-LoRA (QLoRA 4 bits, ~300 images) n'est faite que si le temps le permet.
