"""
Purchase Value Prediction - Data Processing Module

This module handles data loading, cleaning, and preprocessing
for the purchase value prediction ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Handles data loading and preprocessing for the ML pipeline."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files."""
        print("Loading training data...")
        train_data = pd.read_csv(train_path)
        
        print("Loading test data...")  
        test_data = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        return train_data, test_data
    
    def basic_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic exploratory data analysis."""
        eda_results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().sort_values(ascending=False),
            'dtypes': df.dtypes.value_counts(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Target statistics (if present)
        if 'purchaseValue' in df.columns:
            target = df['purchaseValue']
            eda_results['target_stats'] = {
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'zeros': (target == 0).sum(),
                'non_zeros': (target > 0).sum()
            }
        
        return eda_results
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess training and test data for ML models."""
        # Separate target from features
        if 'purchaseValue' in train_df.columns:
            y = train_df['purchaseValue'].values
            X_train = train_df.drop('purchaseValue', axis=1)
        else:
            raise ValueError("Target column 'purchaseValue' not found in training data")
            
        X_test = test_df.copy()
        
        # Ensure same columns in train and test
        common_columns = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
        
        # Handle missing values
        X_train = self._handle_missing_values(X_train)
        X_test = self._handle_missing_values(X_test)
        
        # Encode categorical features
        X_train = self._encode_categorical_features(X_train, fit=True)
        X_test = self._encode_categorical_features(X_test, fit=False)
        
        # Ensure same column order
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Scale numerical features  
        X_train = self._scale_features(X_train, fit=True)
        X_test = self._scale_features(X_test, fit=False)
        
        return X_train, X_test, y
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode or 'unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
            
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if fit:
                # Add 'unknown' to the training data to handle unseen categories later
                df[col] = df[col].astype(str).fillna('unknown')
                unique_values = df[col].unique().tolist()
                if 'unknown' not in unique_values:
                    unique_values.append('unknown')
                
                le = LabelEncoder()
                le.fit(unique_values)
                df[col] = le.transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories by mapping them to 'unknown'
                    unique_vals = set(self.label_encoders[col].classes_)
                    df[col] = df[col].astype(str).fillna('unknown').apply(
                        lambda x: x if x in unique_vals else 'unknown'
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    # If encoder doesn't exist, fill with 0
                    df[col] = 0
                    
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale numerical features."""
        if fit:
            scaled_data = self.scaler.fit_transform(df)
            self.feature_names = df.columns.tolist()
        else:
            scaled_data = self.scaler.transform(df)
            
        return scaled_data


def create_train_val_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Create train/validation split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)