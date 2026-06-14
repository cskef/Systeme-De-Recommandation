"""
Script principal pour exécuter le pipeline LightGCN complet
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Ajouter le répertoire src au chemin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import set_seed, get_device, create_directories
from data_loader import MovieLensDataLoader
from graph_builder import BipartiteGraphBuilder
from lightgcn_model import LightGCN
from trainer import LightGCNTrainer
from evaluator import Evaluator
import config


def main():
    """
    Pipeline complet du projet LightGCN
    """
    # Configuration
    parser = argparse.ArgumentParser(description='LightGCN Recommendation System')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--embedding_dim', type=int, default=config.EMBEDDING_DIM)
    parser.add_argument('--n_layers', type=int, default=config.N_LAYERS)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--min_rating', type=float, default=config.MIN_RATING)
    parser.add_argument('--test_split', type=float, default=config.TEST_SPLIT)
    parser.add_argument('--data_dir', type=str, default=str(config.RAW_DATA_DIR))
    parser.add_argument('--config', type=str, choices=['quick_test', 'standard', 'large', 'cpu'], 
                       help='Utiliser une configuration prédéfinie')
    
    args = parser.parse_args()
    
    # Appliquer la configuration prédéfinie si spécifiée
    if args.config:
        pred_config = config.get_config(args.config)
        args.epochs = pred_config['epochs']
        args.batch_size = pred_config['batch_size']
        args.embedding_dim = pred_config['embedding_dim']
        args.n_layers = pred_config['n_layers']
        args.learning_rate = pred_config['learning_rate']
        print(f"\n✓ Configuration prédéfinie appliquée: {args.config}")
    
    # Setup
    print("=" * 80)
    print("LightGCN Recommendation System")
    print("=" * 80)
    
    set_seed(args.seed)
    device = get_device()
    create_directories()
    
    # ============================================================
    # 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
    # ============================================================
    print("\n[1/5] Chargement et prétraitement des données...")
    
    # Vérifier si le dataset exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset non trouvé: {args.data_dir}")
        print("Veuillez télécharger MovieLens ml-latest-small depuis:")
        print("https://grouplens.org/datasets/movielens/")
        sys.exit(1)
    
    loader = MovieLensDataLoader(args.data_dir)
    loader.load_ratings()
    interactions = loader.preprocess(min_rating=args.min_rating)
    
    stats = loader.get_stats()
    print(f"\n  ✓ Statistiques du dataset:")
    for key, value in stats.items():
        print(f"    - {key}: {value}")
    
    loader.save_processed()
    
    # ============================================================
    # 2. CONSTRUCTION DU GRAPHE BIPARTI
    # ============================================================
    print("\n[2/5] Construction du graphe biparti...")
    
    graph_builder = BipartiteGraphBuilder(stats['n_users'], stats['n_items'])
    edge_index = graph_builder.build_from_interactions(interactions)
    edge_index, edge_weight = graph_builder.get_sparse_edge_index()
    graph_builder.save_graph()
    
    print(f"  ✓ Graphe construit avec succès")
    
    # ============================================================
    # 3. SPLITTING TRAIN/TEST
    # ============================================================
    print("\n[3/5] Splitting des données...")
    
    interactions_array = interactions[['user_id', 'item_id']].values
    n_train = int(len(interactions_array) * (1 - args.test_split))
    
    # Mélanger et splitter
    indices = np.random.permutation(len(interactions_array))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_interactions = interactions_array[train_indices]
    test_interactions_array = interactions_array[test_indices]
    
    # Créer un dictionnaire pour les interactions de test
    test_interactions_dict = {}
    for user_id, item_id in test_interactions_array:
        if user_id not in test_interactions_dict:
            test_interactions_dict[user_id] = set()
        test_interactions_dict[user_id].add(item_id)
    
    print(f"  ✓ Train: {len(train_interactions)}, Test: {len(test_interactions_array)}")
    
    # ============================================================
    # 4. ENTRAÎNEMENT DU MODÈLE
    # ============================================================
    print("\n[4/5] Entraînement du modèle LightGCN...")
    
    model = LightGCN(
        n_users=stats['n_users'],
        n_items=stats['n_items'],
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        edge_index=edge_index,
        edge_weight=edge_weight
    )
    model.set_graph(edge_index, edge_weight)
    
    trainer = LightGCNTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    trainer.train(
        train_interactions=train_interactions,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=max(1, args.epochs // 5)
    )
    
    print(f"  ✓ Entraînement terminé")
    
    # ============================================================
    # 5. ÉVALUATION
    # ============================================================
    print("\n[5/5] Évaluation du modèle...")
    
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
    
    metrics = Evaluator.evaluate_batch(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        test_interactions=test_interactions_dict,
        k_values=[10, 20]
    )
    
    print(f"\n  ✓ Résultats d'évaluation:")
    for metric_name, metric_value in metrics.items():
        print(f"    - {metric_name}: {metric_value:.4f}")
    
    # ============================================================
    # VISUALISATIONS
    # ============================================================
    print("\n[BONUS] Génération des visualisations...")
    
    # Courbes de convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trainer.training_history['epoch'], trainer.training_history['train_loss'], 
            label='Training Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LightGCN Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/training_loss.png', dpi=100)
    print("  ✓ Graphique de convergence sauvegardé")
    
    # Métriques en barres
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax.bar(metric_names, metric_values)
    ax.set_ylabel('Score')
    ax.set_title('LightGCN Evaluation Metrics')
    ax.set_ylim([0, 1])
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('results/plots/evaluation_metrics.png', dpi=100)
    print("  ✓ Graphique des métriques sauvegardé")
    
    print("\n" + "=" * 80)
    print("✓ Pipeline LightGCN complété avec succès!")
    print("=" * 80)


if __name__ == '__main__':
    main()
