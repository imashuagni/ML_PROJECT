"""
Purchase Value Prediction - Main Training Pipeline

This script runs the complete ML pipeline for predicting purchase values
from web analytics data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor, create_train_val_split
from model_training import ModelTrainer, create_submission


def main():
    """Main pipeline execution."""
    
    # Define paths
    data_dir = Path('data/raw')
    train_path = data_dir / 'train_data.csv'
    test_path = data_dir / 'test_data.csv'
    sample_submission_path = data_dir / 'sample_submission.csv'
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PURCHASE VALUE PREDICTION PIPELINE")
    print("=" * 60)
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Step 1: Load and explore data
    print("\n1. LOADING AND EXPLORING DATA")
    print("-" * 40)
    
    train_df, test_df = processor.load_data(train_path, test_path)
    
    # Basic EDA
    eda_results = processor.basic_eda(train_df)
    print(f"\nDataset Overview:")
    print(f"Training shape: {eda_results['shape']}")
    print(f"Missing values: {eda_results['missing_values']}")
    print(f"Numeric columns: {len(eda_results['numeric_columns'])}")
    print(f"Categorical columns: {len(eda_results['categorical_columns'])}")
    
    if 'target_stats' in eda_results:
        print(f"\nTarget Variable Statistics:")
        target_stats = eda_results['target_stats']
        print(f"Mean: {target_stats['mean']:.4f}")
        print(f"Std: {target_stats['std']:.4f}")
        print(f"Min: {target_stats['min']:.4f}")
        print(f"Max: {target_stats['max']:.4f}")
        print(f"Zeros: {target_stats['zeros']} ({target_stats['zeros']/len(train_df)*100:.1f}%)")
        print(f"Non-zeros: {target_stats['non_zeros']} ({target_stats['non_zeros']/len(train_df)*100:.1f}%)")
    
    # Step 2: Data preprocessing
    print("\n2. DATA PREPROCESSING")
    print("-" * 40)
    
    X_train_full, X_test, y_train_full = processor.preprocess_data(train_df, test_df)
    print(f"Processed training data shape: {X_train_full.shape}")
    print(f"Processed test data shape: {X_test.shape}")
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Step 3: Model training
    print("\n3. MODEL TRAINING")
    print("-" * 40)
    
    trainer.train_all_models(X_train, y_train)
    
    # Step 4: Model evaluation
    print("\n4. MODEL EVALUATION")
    print("-" * 40)
    
    evaluation_results = trainer.evaluate_all_models(X_val, y_val)
    print("\nValidation Results:")
    print(evaluation_results.round(4))
    
    # Save evaluation results
    evaluation_results.to_csv(results_dir / 'model_evaluation.csv')
    
    # Find best model based on RMSE
    best_model = evaluation_results['rmse'].idxmin()
    print(f"\nBest Model: {best_model}")
    print(f"Best RMSE: {evaluation_results.loc[best_model, 'rmse']:.4f}")
    
    # Step 5: Feature importance (if available)
    print("\n5. FEATURE IMPORTANCE")
    print("-" * 40)
    
    feature_importance = trainer.get_feature_importance(best_model, processor.feature_names)
    if feature_importance is not None:
        print(f"\nTop 10 Most Important Features ({best_model}):")
        print(feature_importance.head(10))
        feature_importance.to_csv(results_dir / f'{best_model}_feature_importance.csv', index=False)
    
    # Step 6: Final predictions and submission
    print("\n6. GENERATING FINAL PREDICTIONS")
    print("-" * 40)
    
    # Retrain best model on full training data
    trainer.train_model(best_model, X_train_full, y_train_full)
    
    # Make predictions on test set
    test_predictions = trainer.predict(best_model, X_test)
    
    # Create submission file
    submission_path = results_dir / 'submission.csv'
    create_submission(test_predictions, sample_submission_path, submission_path)
    
    # Save best model
    model_path = models_dir / f'best_model_{best_model}.pkl'
    trainer.save_model(best_model, model_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best model: {best_model}")
    print(f"Validation RMSE: {evaluation_results.loc[best_model, 'rmse']:.4f}")
    print(f"Submission file: {submission_path}")
    print(f"Model saved: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()