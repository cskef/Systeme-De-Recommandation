"""
Module de chargement et prétraitement du dataset MovieLens
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class MovieLensDataLoader:
    """
    Loader pour le dataset MovieLens ml-latest-small
    """
    
    def __init__(self, data_dir: str = 'data/raw/ml-latest-small'):
        """
        Initialiser le loader
        
        Args:
            data_dir: Chemin du répertoire contenant les données MovieLens
        """
        self.data_dir = data_dir
        self.ratings = None
        self.movies = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
    
    def load_ratings(self):
        """
        Charger le fichier ratings.csv
        
        Returns:
            DataFrame des ratings
        """
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Fichier non trouvé: {ratings_path}")
        
        self.ratings = pd.read_csv(ratings_path)
        print(f"Ratings chargés: {self.ratings.shape[0]} interactions")
        print(f"Utilisateurs uniques: {self.ratings['userId'].nunique()}")
        print(f"Films uniques: {self.ratings['movieId'].nunique()}")
        return self.ratings
    
    def load_movies(self):
        """
        Charger le fichier movies.csv
        
        Returns:
            DataFrame des films
        """
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Fichier non trouvé: {movies_path}")
        
        self.movies = pd.read_csv(movies_path)
        print(f"Films chargés: {self.movies.shape[0]} films")
        return self.movies
    
    def preprocess(self, min_rating: float = 3.5):
        """
        Prétraiter les données:
        - Filtrer les ratings bas (considérer comme pas d'intérêt)
        - Encoder les utilisateurs et films
        
        Args:
            min_rating: Seuil de rating minimum pour garder une interaction
        
        Returns:
            DataFrame prétraité
        """
        if self.ratings is None:
            self.load_ratings()
        
        # Filtrer les ratings bas (garder seulement les interactions positives)
        interactions = self.ratings[self.ratings['rating'] >= min_rating].copy()
        print(f"Interactions après filtrage (rating >= {min_rating}): {len(interactions)}")
        
        # Encoder les utilisateurs et films
        interactions['user_id'] = self.user_encoder.fit_transform(interactions['userId'])
        interactions['item_id'] = self.movie_encoder.fit_transform(interactions['movieId'])
        
        # Supprimer les colonnes inutiles
        interactions = interactions[['user_id', 'item_id', 'rating', 'timestamp']]
        
        self.ratings = interactions
        return interactions
    
    def get_stats(self):
        """
        Obtenir les statistiques du dataset
        
        Returns:
            dict avec les statistiques
        """
        if self.ratings is None:
            raise ValueError("Données non chargées. Appelez load_ratings() et preprocess() d'abord.")
        
        stats = {
            'n_users': self.ratings['user_id'].nunique(),
            'n_items': self.ratings['item_id'].nunique(),
            'n_interactions': len(self.ratings),
            'density': len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['item_id'].nunique()),
        }
        return stats
    
    def save_processed(self, output_dir: str = 'data/processed'):
        """
        Sauvegarder les données prétraitées
        
        Args:
            output_dir: Répertoire de sortie
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'interactions.csv')
        self.ratings.to_csv(output_path, index=False)
        print(f"Interactions sauvegardées: {output_path}")
        
        # Sauvegarder les encodeurs
        np.save(os.path.join(output_dir, 'user_encoder.npy'), 
                self.user_encoder.classes_)
        np.save(os.path.join(output_dir, 'movie_encoder.npy'), 
                self.movie_encoder.classes_)
