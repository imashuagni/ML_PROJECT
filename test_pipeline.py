"""
Quick test of the ML pipeline with a small sample of data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor, create_train_val_split
from model_training import ModelTrainer

def test_pipeline():
    """Test the ML pipeline with a smaller dataset."""
    
    print("=" * 50)
    print("TESTING ML PIPELINE")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/raw/train_data.csv')
    test_df = pd.read_csv('data/raw/test_data.csv')
    
    # Use a smaller sample for testing
    print("Sampling data for quick test...")
    train_sample = train_df.sample(n=5000, random_state=42)
    test_sample = test_df.sample(n=1000, random_state=42)
    
    print(f"Using {len(train_sample)} training samples and {len(test_sample)} test samples")
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_full, X_test, y_train_full = processor.preprocess_data(train_sample, test_sample)
    
    print(f"Processed shapes: X_train={X_train_full.shape}, X_test={X_test.shape}, y_train={y_train_full.shape}")
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"Split shapes: X_train={X_train.shape}, X_val={X_val.shape}")
    
    # Train a simple model
    print("Training model...")
    trainer.train_model('linear_regression', X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate_model('linear_regression', X_val, y_val)
    print(f"Validation metrics: {metrics}")
    
    # Make predictions
    print("Making predictions...")
    predictions = trainer.predict('linear_regression', X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction stats: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    
    print("\n" + "=" * 50)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_pipeline()