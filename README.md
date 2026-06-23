# Classification de maladies des plantes par Deep Learning

Projet académique réalisé à l'**EFREI Paris**, comparaison de 4 architectures de réseaux de neurones pour détecter les maladies des plantes à partir de photos de feuilles.

---

## Contexte

Les maladies agricoles causent chaque année des pertes considérables. Ce projet explore comment le Deep Learning peut automatiser leur détection visuelle, en comparant différentes approches du réseau dense basique jusqu'au transfer learning sur un modèle pré-entraîné sur ImageNet.

---

## Dataset

**PlantVillage**: 10 classes sélectionnées parmi 38 (5 plantes, chacune saine et malade)

| Plante | Classe saine | Classe malade |
|--------|-------------|---------------|
| Pomme | Apple healthy | Apple scab |
| Maïs | Corn healthy | Common rust |
| Raisin | Grape healthy | Black rot |
| Pomme de terre | Potato healthy | Late blight |
| Tomate | Tomato healthy | Bacterial spot |

> ~4 000 images, redimensionnées à 128×128 px découpage 70% train / 15% val / 15% test.

---

## Modèles comparés

### 1. MLP dense
Réseau entièrement connecté sert de baseline. Aplatit l'image et enchaîne deux blocs Dense.

### 2. CNN simple
Trois blocs Conv2D + MaxPooling sans régularisation met en évidence le surapprentissage.

### 3. CNN régularisé
Même architecture, avec toute la panoplie de régularisation :
- Data augmentation (flip, rotation, zoom, contraste)
- BatchNormalization après chaque convolution
- Dropout à chaque bloc
- Régularisation ElasticNet (L1 + L2) sur les couches denses
- GlobalAveragePooling au lieu de Flatten

### 4. ResNet maison
Réseau résiduel construit from scratch avec des skip connections évite la disparition du gradient dans les réseaux profonds.

### 5. Transfer Learning — DenseNet121
DenseNet121 pré-entraîné sur ImageNet (gelé), suivi d'un fine-tuning des 30 dernières couches.

---

## Techniques de régularisation

| Technique | Rôle |
|-----------|------|
| Data augmentation | Empêche la mémorisation des images |
| BatchNormalization | Stabilise et accélère l'entraînement |
| Dropout | Force le réseau à ne pas dépendre de neurones individuels |
| ElasticNet (L1+L2) | Pénalise les poids trop grands |
| Initialisation He / Glorot | Démarre l'entraînement avec de bons poids |
| GlobalAveragePooling | Réduit drastiquement le nombre de paramètres |
| EarlyStopping | Sauvegarde le meilleur modèle automatiquement |
| ReduceLROnPlateau | Affine la convergence en fin d'entraînement |

---

## Résultats

| Modèle | Accuracy test | F1 macro |
|--------|:-------------:|:--------:|
| MLP dense | ~70% | ~0.70 |
| CNN simple | ~97% | ~0.97 |
| CNN régularisé | ~94% | ~0.94 |
| ResNet maison | ~96% | ~0.96 |
| **Transfer Learning DenseNet121** | **~98%** | **~0.98** |

> Le CNN simple surfit légèrement (train ≈ 100%) mais performe bien sur ce dataset de taille modeste.
> Le transfer learning atteint les meilleures performances avec le moins d'époques d'entraînement.

---

## Stack technique

- **Python 3.10**
- **TensorFlow 2.10 / Keras** construction et entraînement des modèles
- **NumPy / Pandas** manipulation des données
- **Matplotlib / Seaborn** visualisation
- **scikit-learn** métriques (F1, matrice de confusion)
- **GPU** NVIDIA RTX 3050, entraînement accéléré via CUDA 11.8

---

## Structure du projet

```
projet-plant/
├── data/
│   └── PlantVillage/          # Dataset (non versionné)
├── models/
│   ├── best_mlp.keras
│   ├── best_cnn_simple.keras
│   ├── best_cnn_reg.keras
│   ├── best_resnet.keras
│   ├── best_transfer.keras
│   └── modele_final.keras     # Meilleur modèle sauvegardé
├── Plant_Disease_Local_Keras.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Cloner le repo
git clone <url-du-repo>
cd projet-plant

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset PlantVillage depuis Kaggle
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Placer le dossier `color/` sous data/PlantVillage/
```

---

## Lancer le notebook

Ouvrir `Plant_Disease_Local_Keras.ipynb` dans VS Code ou Jupyter et sélectionner le kernel **Python 3.10 (venv)**.

---

## Ce que j'ai appris

- Comprendre l'impact concret de chaque technique de régularisation sur les courbes d'entraînement
- Construire un ResNet from scratch en comprenant le rôle des skip connections
- Mettre en place du transfer learning en deux phases (feature extraction + fine-tuning)
- Configurer un environnement Deep Learning sous Windows avec GPU (CUDA + cuDNN + TF)
