"""
Package LightGCN - Système de Recommandation basé sur les Graphes
"""

__version__ = "1.0.0"
__author__ = "Étudiant Master IA"

from .utils import set_seed, get_device, create_directories
from .data_loader import MovieLensDataLoader
from .graph_builder import BipartiteGraphBuilder
from .lightgcn_model import LightGCN
from .trainer import LightGCNTrainer, BPRLoss
from .evaluator import Evaluator

__all__ = [
    'set_seed',
    'get_device',
    'create_directories',
    'MovieLensDataLoader',
    'BipartiteGraphBuilder',
    'LightGCN',
    'LightGCNTrainer',
    'BPRLoss',
    'Evaluator',
]
