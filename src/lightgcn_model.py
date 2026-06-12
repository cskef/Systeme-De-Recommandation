"""
Modèle LightGCN pour les systèmes de recommandation
"""
import torch
import torch.nn as nn
import numpy as np


class LightGCN(nn.Module):
    """
    Implémentation de LightGCN (Light Graph Convolutional Network)
    
    Architecture:
    - Couches de propagation de messages sur le graphe biparti
    - Pas de transformations de features, uniquement de la propagation
    - Embeddings apprenables pour les utilisateurs et articles
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 n_layers: int = 3, edge_index=None, edge_weight=None):
        """
        Initialiser le modèle LightGCN
        
        Args:
            n_users: Nombre d'utilisateurs
            n_items: Nombre d'articles
            embedding_dim: Dimension des embeddings
            n_layers: Nombre de couches de convolution
            edge_index: Indices des arêtes du graphe [2, n_edges]
            edge_weight: Poids des arêtes
        """
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initialiser les embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialiser avec une distribution uniforme petite
        nn.init.uniform_(self.user_embedding.weight, a=-0.5/embedding_dim, b=0.5/embedding_dim)
        nn.init.uniform_(self.item_embedding.weight, a=-0.5/embedding_dim, b=0.5/embedding_dim)
        
        # Stocker les informations du graphe
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.adj_matrix = None
    
    def set_graph(self, edge_index, edge_weight=None):
        """
        Définir le graphe
        
        Args:
            edge_index: Indices des arêtes [2, n_edges]
            edge_weight: Poids des arêtes (optionnel)
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self._build_adj_matrix()
    
    def _build_adj_matrix(self):
        """
        Construire la matrice d'adjacence à partir du edge_index
        """
        device = self.user_embedding.weight.device
        n_total = self.n_users + self.n_items
        
        # Récupérer les indices
        row = self.edge_index[0].to(device)
        col = self.edge_index[1].to(device)
        
        # Créer les valeurs (1 ou poids)
        if self.edge_weight is not None:
            value = self.edge_weight.to(device)
        else:
            value = torch.ones(row.shape[0], device=device)
        
        # Calculer le degré inverse pour la normalisation (D^-1/2)
        indices = torch.stack([row, col])
        
        # Construire la matrice d'adjacence normalisée
        # A = D^-1/2 * A * D^-1/2
        row_sum = torch.zeros(n_total, device=device)
        row_sum.scatter_add_(0, row, value)
        row_sum.scatter_add_(0, col, value)
        
        # Inverser et prendre la racine carrée
        row_sum = torch.clamp(row_sum, min=1)
        deg_inv_sqrt = torch.pow(row_sum, -0.5)
        
        # Normaliser les valeurs
        norm_value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
        
        self.adj_matrix = torch.sparse_coo_tensor(
            indices, norm_value, (n_total, n_total), device=device
        ).coalesce()
    
    def forward(self, user_indices=None, item_indices=None):
        """
        Forward pass
        
        Args:
            user_indices: Indices des utilisateurs (optionnel, sinon retourner tous)
            item_indices: Indices des articles (optionnel, sinon retourner tous)
        
        Returns:
            Embeddings des utilisateurs et articles après propagation
        """
        # Récupérer les embeddings initiaux
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concaténer tous les embeddings (utilisateurs + articles)
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Listes pour stocker les embeddings à chaque couche
        embs = [all_emb]
        
        # Propager le signal à travers les couches
        for _ in range(self.n_layers):
            # Multiplier par la matrice d'adjacence normalisée
            if self.adj_matrix is None:
                self._build_adj_matrix()
            
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs.append(all_emb)
        
        # Moyenne des embeddings de toutes les couches (agrégation)
        final_emb = sum(embs) / len(embs)
        
        # Séparer les embeddings des utilisateurs et articles
        user_emb_final = final_emb[:self.n_users]
        item_emb_final = final_emb[self.n_users:]
        
        if user_indices is None:
            user_indices = torch.arange(self.n_users, device=user_emb_final.device)
        if item_indices is None:
            item_indices = torch.arange(self.n_items, device=item_emb_final.device)
        
        return user_emb_final[user_indices], item_emb_final[item_indices]
    
    def get_embeddings(self):
        """
        Obtenir les embeddings finaux de tous les utilisateurs et articles
        
        Returns:
            Tuple (user_embeddings, item_embeddings)
        """
        user_emb, item_emb = self.forward()
        return user_emb.detach(), item_emb.detach()
