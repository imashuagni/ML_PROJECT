"""
Purchase Value Prediction - Model Training Module

This module contains different ML models and training utilities
for the purchase value prediction task.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'linear_regression': {
                'model_class': LinearRegression,
                'params': {}
            },
            'random_forest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model_class': XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
            
        # Add LightGBM if available  
        if LIGHTGBM_AVAILABLE:
            self.model_configs['lightgbm'] = {
                'model_class': LGBMRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train a specific model."""
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Model {model_name} not available. Choose from: {available_models}")
        
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])
        
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        self.models[model_name] = model
        print(f"{model_name} training completed.")
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train all available models."""
        for model_name in self.model_configs.keys():
            self.train_model(model_name, X_train, y_train)
    
    def evaluate_model(self, model_name: str, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model on validation data."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.models[model_name]
        y_pred = model.predict(X_val)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """Evaluate all trained models and return results as DataFrame."""
        results = []
        
        for model_name in self.models.keys():
            metrics = self.evaluate_model(model_name, X_val, y_val)
            metrics['model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results).set_index('model')
    
    def predict(self, model_name: str, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        return self.models[model_name].predict(X_test)
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """Load a trained model from disk."""
        self.models[model_name] = joblib.load(filepath)
        print(f"Model {model_name} loaded from {filepath}")
    
    def get_feature_importance(self, model_name: str, feature_names: list = None) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} doesn't support feature importance.")
            return None


def create_submission(predictions: np.ndarray, sample_submission_path: str, output_path: str) -> None:
    """Create submission file from predictions."""
    sample_df = pd.read_csv(sample_submission_path)
    
    submission_df = sample_df.copy()
    submission_df['purchaseValue'] = predictions
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Show summary statistics
    print(f"Prediction statistics:")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std: {predictions.std():.4f}")  
    print(f"Min: {predictions.min():.4f}")
    print(f"Max: {predictions.max():.4f}")
    print(f"Zeros: {(predictions == 0).sum()}")
    print(f"Non-zeros: {(predictions > 0).sum()}")