"""
Configuration du projet LightGCN
Ce fichier centralise tous les paramètres du projet
"""

import os
from pathlib import Path


# ===========================
# CHEMINS
# ===========================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'ml-latest-small'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = RESULTS_DIR / 'logs'
MODELS_DIR = RESULTS_DIR / 'models'
PLOTS_DIR = RESULTS_DIR / 'plots'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
SRC_DIR = PROJECT_ROOT / 'src'

# Créer les répertoires s'ils n'existent pas
for dir_path in [PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ===========================
# DONNÉES
# ===========================
DATASET_URL = "https://grouplens.org/datasets/movielens/"
RATINGS_FILE = RAW_DATA_DIR / 'ratings.csv'
MOVIES_FILE = RAW_DATA_DIR / 'movies.csv'
LINKS_FILE = RAW_DATA_DIR / 'links.csv'

# Preprocessing
MIN_RATING = 3.5  # Seuil pour filtrer les ratings bas
TEST_SPLIT = 0.2  # Proportion de données de test


# ===========================
# MODÈLE
# ===========================
# Architecture
EMBEDDING_DIM = 64
N_LAYERS = 3  # Nombre de couches GCN

# Entraînement
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Régularisation
L2_REG = 1e-5

# Optimiseur
OPTIMIZER = 'adam'  # 'adam', 'sgd', 'adamw'


# ===========================
# ÉVALUATION
# ===========================
# Métriques
EVAL_K_VALUES = [10, 20]  # K pour Recall@K et NDCG@K
EVAL_METRICS = ['recall', 'ndcg', 'precision']

# Checkpoints
SAVE_CHECKPOINT_EVERY = 10  # Sauvegarder tous les N epochs
KEEP_TOP_K_MODELS = 3  # Garder les 3 meilleurs modèles


# ===========================
# RESSOURCES
# ===========================
SEED = 42
USE_GPU = True
NUM_WORKERS = 4  # Pour les data loaders


# ===========================
# LOGS & VISUALISATIONS
# ===========================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Visualisations
PLOT_TRAINING_LOSS = True
PLOT_METRICS = True
PLOT_EMBEDDINGS = False  # Réduction dimensionalité pour visualiser


# ===========================
# HYPERPARAMÈTRES AVANCÉS
# ===========================

# BPR Loss
NEGATIVE_SAMPLING_STRATEGY = 'random'  # 'random', 'importance'

# Scheduler (optionnel)
USE_SCHEDULER = False
SCHEDULER_TYPE = 'cosine'  # 'linear', 'cosine', 'exponential'
SCHEDULER_STEP_SIZE = 10

# Early stopping (optionnel)
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_METRIC = 'ndcg@10'


# ===========================
# CONFIGURATIONS PRÉDÉFINIES
# ===========================
CONFIGS = {
    'quick_test': {
        'epochs': 5,
        'batch_size': 256,
        'embedding_dim': 32,
        'n_layers': 2,
        'learning_rate': 0.001,
    },
    'standard': {
        'epochs': 100,
        'batch_size': 1024,
        'embedding_dim': 64,
        'n_layers': 3,
        'learning_rate': 0.001,
    },
    'large': {
        'epochs': 200,
        'batch_size': 2048,
        'embedding_dim': 128,
        'n_layers': 4,
        'learning_rate': 0.0005,
    },
    'cpu': {
        'epochs': 50,
        'batch_size': 512,
        'embedding_dim': 32,
        'n_layers': 2,
        'learning_rate': 0.001,
    },
}


# ===========================
# FONCTIONS UTILITAIRES
# ===========================
def get_config(config_name: str = 'standard') -> dict:
    """
    Obtenir une configuration prédéfinie
    
    Args:
        config_name: Nom de la configuration
    
    Returns:
        Dictionnaire de configuration
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Configuration inconnue: {config_name}")
    return CONFIGS[config_name].copy()


def print_config():
    """Afficher la configuration actuelle"""
    print("\n" + "="*60)
    print("CONFIGURATION DU PROJET")
    print("="*60)
    
    print(f"\n📂 CHEMINS:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    print(f"\n🧠 MODÈLE:")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  N Layers: {N_LAYERS}")
    
    print(f"\n⚙️ ENTRAÎNEMENT:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    
    print(f"\n📊 ÉVALUATION:")
    print(f"  K Values: {EVAL_K_VALUES}")
    print(f"  Metrics: {EVAL_METRICS}")
    
    print(f"\n🔧 RESSOURCES:")
    print(f"  Seed: {SEED}")
    print(f"  Use GPU: {USE_GPU}")
    print(f"  Num Workers: {NUM_WORKERS}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    print_config()
