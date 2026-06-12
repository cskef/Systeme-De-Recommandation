"""
Module d'entraînement pour LightGCN
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    Optimise le ranking des items positifs vs négatifs
    """
    
    def __init__(self, weight_decay: float = 1e-5):
        """
        Args:
            weight_decay: Poids de la régularisation L2
        """
        super(BPRLoss, self).__init__()
        self.weight_decay = weight_decay
    
    def forward(self, user_emb: torch.Tensor, pos_item_emb: torch.Tensor,
                neg_item_emb: torch.Tensor, model=None) -> torch.Tensor:
        """
        Calculer la perte BPR
        
        Args:
            user_emb: Embeddings des utilisateurs [batch_size, embedding_dim]
            pos_item_emb: Embeddings des articles positifs [batch_size, embedding_dim]
            neg_item_emb: Embeddings des articles négatifs [batch_size, embedding_dim]
            model: Modèle (pour la régularisation)
        
        Returns:
            Valeur de loss scalaire
        """
        # Scores pour les articles positifs et négatifs
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        # BPR Loss: -log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        # Ajouter la régularisation L2
        if model is not None:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.sum(param ** 2) / 2
            loss += self.weight_decay * l2_loss
        
        return loss


class LightGCNTrainer:
    """
    Entraîneur pour le modèle LightGCN
    """
    
    def __init__(self, model, device='cpu', learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, log_dir: str = 'results/logs'):
        """
        Initialiser l'entraîneur
        
        Args:
            model: Modèle LightGCN
            device: Device (cpu ou cuda)
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids (régularisation L2)
            log_dir: Répertoire des logs
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=learning_rate, 
                                         weight_decay=weight_decay)
        self.loss_fn = BPRLoss(weight_decay=weight_decay)
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_metrics': []
        }
    
    def _sample_negative_items(self, positive_items: np.ndarray, 
                               n_items: int, batch_size: int) -> np.ndarray:
        """
        Échantillonner des articles négatifs
        
        Args:
            positive_items: Indices des articles positifs
            n_items: Nombre total d'articles
            batch_size: Taille du batch
        
        Returns:
            Indices des articles négatifs
        """
        negative_items = np.random.randint(0, n_items, batch_size)
        
        # S'assurer que les articles négatifs ne sont pas dans les positifs
        for i in range(batch_size):
            while negative_items[i] in positive_items:
                negative_items[i] = np.random.randint(0, n_items)
        
        return negative_items
    
    def train_epoch(self, train_interactions: list, batch_size: int = 1024):
        """
        Entraîner une époque
        
        Args:
            train_interactions: Liste des interactions [user_id, item_id]
            batch_size: Taille des batches
        
        Returns:
            Perte moyenne de l'époque
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Créer des batches aléatoires
        n_interactions = len(train_interactions)
        indices = np.random.permutation(n_interactions)
        
        with tqdm(total=n_interactions, desc="Training", leave=False) as pbar:
            for i in range(0, n_interactions, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_interactions = train_interactions[batch_indices]
                
                # Récupérer les utilisateurs et articles positifs
                user_ids = torch.LongTensor(batch_interactions[:, 0]).to(self.device)
                pos_item_ids = torch.LongTensor(batch_interactions[:, 1]).to(self.device)
                
                # Échantillonner les articles négatifs
                neg_item_ids = self._sample_negative_items(
                    batch_interactions[:, 1], self.model.n_items, len(user_ids)
                )
                neg_item_ids = torch.LongTensor(neg_item_ids).to(self.device)
                
                # Forward pass
                user_emb, _ = self.model.forward(user_ids, pos_item_ids)
                pos_item_emb = self.model.item_embedding(pos_item_ids)
                neg_item_emb = self.model.item_embedding(neg_item_ids)
                
                # Calculer la perte
                loss = self.loss_fn(user_emb, pos_item_emb, neg_item_emb, self.model)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                pbar.update(len(batch_interactions))
        
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
    
    def train(self, train_interactions: list, n_epochs: int = 100,
              batch_size: int = 1024, save_every: int = 10,
              model_dir: str = 'results/models'):
        """
        Entraîner le modèle pour plusieurs épocités
        
        Args:
            train_interactions: Liste des interactions
            n_epochs: Nombre d'épocités
            batch_size: Taille des batches
            save_every: Sauvegarder tous les N epochs
            model_dir: Répertoire de sauvegarde des modèles
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')
        
        with open(log_file, 'w') as f:
            f.write(f"LightGCN Training Log - {timestamp}\n")
            f.write(f"Epochs: {n_epochs}, Batch size: {batch_size}\n")
            f.write("=" * 50 + "\n\n")
        
        for epoch in range(n_epochs):
            # Entraînement
            train_loss = self.train_epoch(train_interactions, batch_size)
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            
            # Log
            log_msg = f"Epoch [{epoch+1}/{n_epochs}] - Loss: {train_loss:.4f}"
            print(log_msg)
            
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")
            
            # Sauvegarder le modèle
            if (epoch + 1) % save_every == 0:
                model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f"Modèle sauvegardé: {model_path}")
        
        # Sauvegarder le modèle final
        final_model_path = os.path.join(model_dir, 'model_final.pt')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Modèle final sauvegardé: {final_model_path}")
        
        print(f"Logs sauvegardés: {log_file}")
