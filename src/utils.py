"""
Utilitaires et helpers pour le projet LightGCN
"""
import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Fixer tous les seeds pour la reproductibilité
    
    Args:
        seed: Valeur de seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Retourner le device optimal (GPU ou CPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU non disponible, utilisation du CPU")
    return device


def create_directories():
    """
    Créer les répertoires nécessaires s'ils n'existent pas
    """
    dirs = [
        'data/raw',
        'data/processed',
        'results/logs',
        'results/models',
        'results/plots'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Sauvegarder un checkpoint du modèle
    
    Args:
        model: Modèle PyTorch
        optimizer: Optimiseur
        epoch: Numéro d'époque
        loss: Valeur de loss
        path: Chemin de sauvegarde
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint sauvegardé: {path}")


def load_checkpoint(model, optimizer, path):
    """
    Charger un checkpoint du modèle
    
    Args:
        model: Modèle PyTorch
        optimizer: Optimiseur
        path: Chemin du checkpoint
    
    Returns:
        Numéro d'époque et valeur de loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint chargé: {path}")
    return epoch, loss
