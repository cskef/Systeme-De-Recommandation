"""
Module de construction du graphe biparti utilisateur-article
"""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from pathlib import Path


class BipartiteGraphBuilder:
    """
    Constructeur de graphe biparti pour les interactions utilisateur-article
    """
    
    def __init__(self, n_users: int, n_items: int):
        """
        Initialiser le builder
        
        Args:
            n_users: Nombre d'utilisateurs
            n_items: Nombre d'articles
        """
        self.n_users = n_users
        self.n_items = n_items
        self.edge_index = None
        self.edge_weight = None
    
    def build_from_interactions(self, interactions_df):
        """
        Construire le graphe à partir des interactions
        
        Args:
            interactions_df: DataFrame avec colonnes ['user_id', 'item_id', 'rating']
        
        Returns:
            Tenseur edge_index [2, n_edges]
        """
        # Récupérer les indices des utilisateurs et articles
        user_indices = interactions_df['user_id'].values
        item_indices = interactions_df['item_id'].values + self.n_users
        
        # Créer les arêtes du graphe biparti (utilisateur -> article et article -> utilisateur)
        # Pour un graphe biparti, on ajoute les arêtes dans les deux directions
        edge_index_forward = np.array([user_indices, item_indices])
        edge_index_backward = np.array([item_indices, user_indices])
        
        # Combiner les deux directions
        self.edge_index = np.concatenate([edge_index_forward, edge_index_backward], axis=1)
        
        # Créer les poids des arêtes (ratings normalisés)
        edge_weight_forward = interactions_df['rating'].values / interactions_df['rating'].max()
        edge_weight_backward = edge_weight_forward.copy()
        self.edge_weight = np.concatenate([edge_weight_forward, edge_weight_backward])
        
        print(f"Graphe construit: {self.edge_index.shape[1]} arêtes")
        print(f"Densité du graphe: {self.edge_index.shape[1] / (self.n_users * self.n_items):.4f}")
        
        return torch.LongTensor(self.edge_index)
    
    def get_adj_matrix(self):
        """
        Obtenir la matrice d'adjacence du graphe complet (utilisateurs + articles)
        
        Returns:
            Matrice d'adjacence sparse au format COO
        """
        n_total = self.n_users + self.n_items
        
        # Créer la matrice d'adjacence
        row = self.edge_index[0]
        col = self.edge_index[1]
        data = self.edge_weight
        
        adj = coo_matrix((data, (row, col)), shape=(n_total, n_total))
        return adj
    
    def get_sparse_edge_index(self):
        """
        Obtenir l'edge_index au format sparse PyTorch
        
        Returns:
            Tenseur sparse avec indices et valeurs
        """
        edge_index_tensor = torch.LongTensor(self.edge_index)
        edge_weight_tensor = torch.FloatTensor(self.edge_weight)
        
        return edge_index_tensor, edge_weight_tensor
    
    def save_graph(self, output_path: str = 'data/processed/graph.pt'):
        """
        Sauvegarder le graphe
        
        Args:
            output_path: Chemin de sauvegarde
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        graph_data = {
            'edge_index': torch.LongTensor(self.edge_index),
            'edge_weight': torch.FloatTensor(self.edge_weight),
            'n_users': self.n_users,
            'n_items': self.n_items,
        }
        torch.save(graph_data, output_path)
        print(f"Graphe sauvegardé: {output_path}")
    
    @staticmethod
    def load_graph(path: str = 'data/processed/graph.pt'):
        """
        Charger un graphe sauvegardé
        
        Args:
            path: Chemin du fichier
        
        Returns:
            Dictionnaire avec les données du graphe
        """
        graph_data = torch.load(path)
        print(f"Graphe chargé: {path}")
        return graph_data
