# Guide Kaggle pour Anti-BAD Challenge

## Étape 1 : Créer un Notebook Kaggle

1. Aller sur https://www.kaggle.com/code
2. Créer un nouveau notebook
3. Activer GPU : Settings → Accelerator → GPU T4 x2 (gratuit)

## Étape 2 : Cloner votre dépôt Git (RECOMMANDÉ)

Dans la première cellule du notebook Kaggle :

```python
# Cloner le repo
!git clone https://github.com/isalinefsy/anti-bad-challenge.git
%cd anti-bad-challenge

# Installer les dépendances
!pip install -q -r requirements.txt

# Télécharger les ressources (modèles + données)
!python download_resources.py
```

**Avantages** :
- ✅ Pas besoin d'uploader manuellement
- ✅ Toujours à jour avec votre code
- ✅ Facile à partager et reproduire

## Étape 2 (Alternative) : Upload manuel

### Option A : Upload direct
- Zipper votre dossier local après `download_resources.py`
- Uploader comme dataset Kaggle

### Option B : Dataset Kaggle
```python
# Dans le notebook Kaggle
!pip install -r /kaggle/input/your-dataset/requirements.txt
!python /kaggle/input/your-dataset/download_resources.py
```

## Étape 3 : Logging des activations

Kaggle offre :
- 30h/semaine GPU gratuit
- Persistent datasets
- Export facile des résultats
