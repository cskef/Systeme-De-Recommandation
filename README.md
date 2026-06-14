# Système de Recommandation basé sur LightGCN

## 📋 Vue d'ensemble

Ce projet implémente un système complet de recommandation basé sur **LightGCN** (Light Graph Convolutional Network), une architecture moderne pour les systèmes de recommandation utilisant les graphes bipartis utilisateur-article.

### Objectifs du projet
- ✅ Charger et prétraiter le dataset MovieLens
- ✅ Construire un graphe biparti des interactions utilisateur-article
- ✅ Implémenter l'architecture LightGCN avec propagation de messages
- ✅ Entraîner le modèle avec la perte BPR (Bayesian Personalized Ranking)
- ✅ Évaluer avec les métriques Recall@K et NDCG@K
- ✅ Générer des recommandations Top-K

---

## 🏗️ Architecture du Projet

```
projet_lightgcn/
├── data/
│   ├── raw/
│   │   └── ml-latest-small/        # Dataset MovieLens (à télécharger)
│   └── processed/
│       ├── interactions.csv
│       ├── graph.pt
│       └── encoders.npy
├── src/
│   ├── data_loader.py              # Chargement du dataset
│   ├── graph_builder.py            # Construction du graphe biparti
│   ├── lightgcn_model.py           # Architecture LightGCN
│   ├── trainer.py                  # Boucle d'entraînement
│   ├── evaluator.py                # Métriques d'évaluation
│   └── utils.py                    # Utilitaires
├── results/
│   ├── logs/                       # Fichiers de logs d'entraînement
│   ├── models/                     # Checkpoints du modèle
│   └── plots/                      # Graphiques de résultats
├── notebooks/
│   └── analysis.ipynb              # Analyse et visualisation
├── main.py                         # Script principal
├── config.py                       # Config du projet
├── requirements.txt                # Dépendances Python
└── README.md                       # Cette documentation
```

---

## 🚀 Démarrage rapide

### 1. Préparation de l'environnement

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Télécharger le dataset

```bash
# Télécharger MovieLens ml-latest-small
cd data/raw/
wget https://grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cd ../..
```

### 3. Lancer le pipeline complet

```bash
python main.py --epochs 100 --batch_size 1024 --learning_rate 0.001
```

#### Options disponibles:
```
--seed              Seed pour la reproductibilité (défaut: 42)
--embedding_dim     Dimension des embeddings (défaut: 64)
--n_layers          Nombre de couches GCN (défaut: 3)
--epochs            Nombre d'epochs d'entraînement (défaut: 100)
--batch_size        Taille des batches (défaut: 1024)
--learning_rate     Taux d'apprentissage (défaut: 0.001)
--weight_decay      Régularisation L2 (défaut: 1e-5)
--min_rating        Seuil de rating minimum (défaut: 3.5)
--test_split        Proportion de données de test (défaut: 0.2)
--data_dir          Chemin du dataset (défaut: data/raw/ml-latest-small)
```

---

## 📊 Composants Principaux

### 1. **data_loader.py** - Chargement des données
- Charge le dataset MovieLens (ratings.csv, movies.csv)
- Prétraitement: filtrage des ratings bas, encodage des IDs
- Calcul des statistiques du dataset
- Sauvegarde des données prétraitées

**Classe principale:** `MovieLensDataLoader`

```python
loader = MovieLensDataLoader('data/raw/ml-latest-small')
loader.load_ratings()
interactions = loader.preprocess(min_rating=3.5)
stats = loader.get_stats()
loader.save_processed()
```

### 2. **graph_builder.py** - Construction du graphe
- Construit un graphe biparti utilisateur-article
- Implémente la normalisation D^-1/2 * A * D^-1/2
- Supporte les poids d'arêtes basés sur les ratings
- Sauvegarde/chargement du graphe

**Classe principale:** `BipartiteGraphBuilder`

```python
graph_builder = BipartiteGraphBuilder(n_users=610, n_items=9742)
edge_index = graph_builder.build_from_interactions(interactions)
adj_matrix = graph_builder.get_adj_matrix()
```

### 3. **lightgcn_model.py** - Architecture LightGCN
- Implémente le modèle LightGCN avec propagation de messages
- Embeddings apprenables pour utilisateurs et articles
- Agrégation multi-couches (moyenne des représentations)
- Pas de transformations de features (simplification par rapport aux GCN classiques)

**Classe principale:** `LightGCN`

```python
model = LightGCN(
    n_users=610,
    n_items=9742,
    embedding_dim=64,
    n_layers=3,
    edge_index=edge_index,
    edge_weight=edge_weight
)

user_emb, item_emb = model.get_embeddings()
```

### 4. **trainer.py** - Entraînement
- Boucle d'entraînement avec BPR Loss
- Échantillonnage négatif aléatoire
- Logs détaillés de progression
- Sauvegarde de checkpoints réguliers

**Composants:**
- `BPRLoss`: Perte Bayesian Personalized Ranking
- `LightGCNTrainer`: Gestionnaire d'entraînement

```python
trainer = LightGCNTrainer(model, device='cuda')
trainer.train(
    train_interactions=train_interactions,
    n_epochs=100,
    batch_size=1024
)
```

### 5. **evaluator.py** - Évaluation
- Implémente les métriques: Recall@K, NDCG@K, Precision@K
- Évaluation batch sur l'ensemble de test
- Ranking-aware (l'ordre des recommandations compte)

**Métriques:**
- **Recall@K**: Proportion des items pertinents retrouvés
- **NDCG@K**: Qualité du ranking (items pertinents en haut)
- **Precision@K**: Proportion des items recommandés qui sont pertinents

```python
metrics = Evaluator.evaluate_batch(
    user_embeddings=user_emb,
    item_embeddings=item_emb,
    test_interactions=test_dict,
    k_values=[10, 20]
)
```

### 6. **utils.py** - Utilitaires
- Fixation des seeds pour la reproductibilité
- Gestion du device (GPU/CPU)
- Sauvegarde/chargement de checkpoints

---

## 📈 Flux de Données

```
Dataset MovieLens
        ↓
[data_loader.py] → Chargement & Nettoyage
        ↓
Interactions (user_id, item_id, rating)
        ↓
[graph_builder.py] → Graphe biparti
        ↓
Edge Index + Adjacency Matrix
        ↓
[lightgcn_model.py] → Propagation de messages (3 couches)
        ↓
Embeddings (utilisateurs & articles)
        ↓
[evaluator.py] → Calcul des scores (dot product)
        ↓
Ranking Top-K → Recommandations
        ↓
[trainer.py] → Évaluation (Recall, NDCG)
```

---

## 🔧 Configuration Recommandée

### Pour GPU:
```python
--epochs 200
--batch_size 2048
--learning_rate 0.001
--embedding_dim 64
--n_layers 3
```

### Pour CPU:
```python
--epochs 50
--batch_size 512
--learning_rate 0.001
--embedding_dim 32
--n_layers 2
```

---

## 📊 Résultats Attendus

Sur le dataset MovieLens ml-latest-small:

| Métrique | Valeur attendue |
|----------|-----------------|
| Recall@10 | 0.15 - 0.25 |
| Recall@20 | 0.20 - 0.35 |
| NDCG@10 | 0.10 - 0.18 |
| NDCG@20 | 0.12 - 0.20 |

*Les valeurs exactes dépendent de la configuration et du seed*

---

## 🔍 Concepts Clés

### Graph Convolutional Network (GCN)
Réseau de neurones opérant sur les graphes, propageant l'information à travers les arêtes.

### LightGCN
Variante simplifiée des GCN pour les systèmes de recommandation:
- Élimine les transformations de features (W_l)
- Conserve uniquement la propagation de messages
- Plus efficace et meilleur pour le collaborative filtering

### Bayesian Personalized Ranking (BPR)
Perte qui optimise le ranking des items positifs vs négatifs:
```
L_BPR = -log(sigmoid(s_ui - s_uj))
```

### Embeddings Bipartites
Représentations vectorielles dans le même espace pour les utilisateurs et articles,
permettant le calcul direct de compatibilité par produit scalaire.

---

## 🎯 Améliorations Possibles

- [ ] Implémenter le negative sampling stratégique
- [ ] Ajouter la validation croisée
- [ ] Tester d'autres fonctions de perte (Hinge loss, Cross-entropy)
- [ ] Implémenter les couches d'attention
- [ ] Optimisation GPU avec mixed precision
- [ ] Explainabilité des recommandations
- [ ] Tests unitaires complets
- [ ] API REST pour les prédictions

---

## 📚 Ressources

- **LightGCN Paper**: [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
- **MovieLens Dataset**: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
- **PyTorch Documentation**: [https://pytorch.org/](https://pytorch.org/)

---

## 📝 Notes Techniques

### Reproductibilité
Tous les seeds sont fixés pour garantir la reproductibilité:
- Random seed
- NumPy seed
- PyTorch seed
- CUDA seed (si GPU disponible)

### Normalisation du Graphe
La matrice d'adjacence est normalisée selon:
```
A_norm = D^(-1/2) * A * D^(-1/2)
```

### Agrégation Multi-couches
Les embeddings finaux sont la moyenne des représentations de toutes les couches:
```
e_final = mean([e_0, e_1, ..., e_L])
```

---

## 🐛 Dépannage

### Erreur: "Dataset non trouvé"
→ Téléchargez MovieLens ml-latest-small et placez-le dans `data/raw/`

### GPU OutOfMemory
→ Réduisez le `batch_size` ou `embedding_dim`

### Performances faibles
→ Augmentez le nombre d'epochs et essayez différents `learning_rate`

### Slow Training
→ Utilisez GPU (CUDA) ou réduisez la taille du dataset

---

## 📄 Licence

Ce projet est fourni à titre académique. Les données MovieLens sont libres d'usage académique.

---

**Auteur**: Étudiant Master IA  
**Date**: 2026  
**Encadrant**: Professeur d'IA
