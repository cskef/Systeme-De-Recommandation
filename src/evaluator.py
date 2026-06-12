"""
Module d'évaluation - Métriques pour les systèmes de recommandation
"""
import numpy as np
import torch


class Evaluator:
    """
    Évaluateur pour les systèmes de recommandation
    Implémente les métriques Recall@K et NDCG@K
    """
    
    @staticmethod
    def recall_at_k(recommended: np.ndarray, ground_truth: set, k: int = 10) -> float:
        """
        Calculer Recall@K
        
        Recall@K = |recommended@K ∩ ground_truth| / |ground_truth|
        
        Args:
            recommended: Indices des articles recommandés (triés par score)
            ground_truth: Set des articles pertinents pour l'utilisateur
            k: Nombre de recommandations considérées
        
        Returns:
            Score Recall@K (0-1)
        """
        if len(ground_truth) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & ground_truth)
        return hits / len(ground_truth)
    
    @staticmethod
    def ndcg_at_k(recommended: np.ndarray, ground_truth: set, k: int = 10) -> float:
        """
        Calculer NDCG@K (Normalized Discounted Cumulative Gain)
        
        NDCG@K = DCG@K / IDCG@K
        où DCG = sum(rel_i / log2(i+1))
        
        Args:
            recommended: Indices des articles recommandés (triés par score)
            ground_truth: Set des articles pertinents pour l'utilisateur
            k: Nombre de recommandations considérées
        
        Returns:
            Score NDCG@K (0-1)
        """
        if len(ground_truth) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        
        # Calculer DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item_id in enumerate(recommended_k):
            if item_id in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # log2(i+2) car indexation à partir de 0
        
        # Calculer IDCG (Ideal DCG - cas où tous les meilleurs items sont recommandés)
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def precision_at_k(recommended: np.ndarray, ground_truth: set, k: int = 10) -> float:
        """
        Calculer Precision@K
        
        Precision@K = |recommended@K ∩ ground_truth| / K
        
        Args:
            recommended: Indices des articles recommandés
            ground_truth: Set des articles pertinents
            k: Nombre de recommandations
        
        Returns:
            Score Precision@K (0-1)
        """
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & ground_truth)
        return hits / k
    
    @staticmethod
    def evaluate_batch(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor,
                      test_interactions: dict, k_values: list = [10, 20]):
        """
        Évaluer le modèle sur un ensemble de test
        
        Args:
            user_embeddings: Embeddings des utilisateurs [n_users, embedding_dim]
            item_embeddings: Embeddings des articles [n_items, embedding_dim]
            test_interactions: Dict {user_id: set de items pertinents}
            k_values: Liste des valeurs de K à évaluer
        
        Returns:
            Dict avec les métriques moyennes
        """
        device = user_embeddings.device
        
        metrics = {f'recall@{k}': 0.0 for k in k_values}
        metrics.update({f'ndcg@{k}': 0.0 for k in k_values})
        metrics.update({f'precision@{k}': 0.0 for k in k_values})
        
        n_users = 0
        
        for user_id, ground_truth in test_interactions.items():
            if len(ground_truth) == 0:
                continue
            
            # Récupérer l'embedding de l'utilisateur
            if isinstance(user_embeddings, torch.Tensor):
                user_emb = user_embeddings[user_id].unsqueeze(0)  # [1, embedding_dim]
                
                # Calculer les scores avec tous les items (dot product)
                scores = torch.mm(user_emb, item_embeddings.t()).squeeze(0)  # [n_items]
                
                # Trier les items par score décroissant
                _, recommended_items = torch.topk(scores, k=len(item_embeddings), 
                                                  largest=True, sorted=True)
                recommended_items = recommended_items.cpu().numpy()
            else:
                # Si en numpy
                user_emb = user_embeddings[user_id]
                scores = np.dot(user_emb, item_embeddings.T)
                recommended_items = np.argsort(-scores)
            
            # Calculer les métriques pour ce utilisateur
            for k in k_values:
                metrics[f'recall@{k}'] += Evaluator.recall_at_k(
                    recommended_items, ground_truth, k
                )
                metrics[f'ndcg@{k}'] += Evaluator.ndcg_at_k(
                    recommended_items, ground_truth, k
                )
                metrics[f'precision@{k}'] += Evaluator.precision_at_k(
                    recommended_items, ground_truth, k
                )
            
            n_users += 1
        
        # Moyenner les métriques
        if n_users > 0:
            for key in metrics:
                metrics[key] /= n_users
        
        return metrics
