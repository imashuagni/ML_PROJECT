"""
Demo script showing how to use the trained models for prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from model_training import ModelTrainer

def demo_prediction():
    """Demonstrate how to use the trained model for new predictions."""
    
    print("=" * 60)
    print("PURCHASE VALUE PREDICTION - DEMO")
    print("=" * 60)
    
    # Load a sample from the test data
    test_df = pd.read_csv('data/raw/test_data.csv')
    sample_data = test_df.sample(n=10, random_state=42)
    
    print(f"Making predictions for {len(sample_data)} sample records...")
    
    # Initialize components (these would typically be loaded from saved state)
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Load the trained model
    model_path = Path('models/best_model_linear_regression.pkl')
    if model_path.exists():
        trainer.load_model('linear_regression', str(model_path))
        print("Loaded trained linear regression model.")
    else:
        print("No trained model found. Please run the training pipeline first.")
        return
    
    # For demo purposes, we'll train the processor on a small sample
    # In a real scenario, you'd save and load the processor state too
    train_df = pd.read_csv('data/raw/train_data.csv').sample(n=1000, random_state=42)
    _, _, _ = processor.preprocess_data(train_df, sample_data)
    
    # Now preprocess the sample data
    _, X_sample, _ = processor.preprocess_data(train_df, sample_data)
    
    # Make predictions
    predictions = trainer.predict('linear_regression', X_sample)
    
    # Display results
    print(f"\nPrediction Results:")
    print("-" * 40)
    results_df = pd.DataFrame({
        'ID': sample_data.index,
        'Predicted_PurchaseValue': predictions,
        'Browser': sample_data['browser'].values,
        'DeviceType': sample_data['deviceType'].values,
        'UserChannel': sample_data['userChannel'].values
    })
    
    print(results_df.to_string(index=False, float_format='%.2f'))
    
    print(f"\nPrediction Summary:")
    print(f"Mean prediction: ${predictions.mean():,.2f}")
    print(f"Max prediction: ${predictions.max():,.2f}")
    print(f"Min prediction: ${predictions.min():,.2f}")
    print(f"Predictions > $0: {(predictions > 0).sum()}/{len(predictions)}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

def show_project_summary():
    """Show a summary of the project components."""
    
    print("=" * 60)
    print("ML PROJECT SUMMARY")
    print("=" * 60)
    
    # Check file structure
    print("📁 Project Structure:")
    directories = ['data/raw', 'src', 'models', 'results', 'notebooks']
    for dir_path in directories:
        if Path(dir_path).exists():
            files = list(Path(dir_path).glob('*'))
            print(f"  ✓ {dir_path}/ ({len(files)} files)")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
    
    print("\n📊 Dataset Information:")
    if Path('data/raw/train_data.csv').exists():
        train_df = pd.read_csv('data/raw/train_data.csv')
        test_df = pd.read_csv('data/raw/test_data.csv')
        print(f"  Training data: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        print(f"  Test data: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
        print(f"  Target range: ${train_df['purchaseValue'].min():.0f} - ${train_df['purchaseValue'].max():,.0f}")
        print(f"  Zero purchases: {(train_df['purchaseValue'] == 0).mean()*100:.1f}%")
    
    print("\n🤖 Model Performance:")
    if Path('results/model_evaluation.csv').exists():
        eval_df = pd.read_csv('results/model_evaluation.csv')
        print("  Model Comparison (Validation RMSE):")
        for _, row in eval_df.iterrows():
            print(f"    {row['model']}: ${row['rmse']:,.0f}")
        best_model = eval_df.loc[eval_df['rmse'].idxmin(), 'model']
        print(f"  Best model: {best_model}")
    
    print("\n📁 Available Files:")
    key_files = [
        'train_pipeline.py',
        'src/data_processing.py', 
        'src/model_training.py',
        'notebooks/data_exploration.ipynb',
        'requirements.txt'
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
    
    print("\n🚀 Quick Start Commands:")
    print("  pip install -r requirements.txt")
    print("  python train_pipeline.py")
    print("  jupyter notebook notebooks/data_exploration.ipynb")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Project Demo and Utilities')
    parser.add_argument('--demo', action='store_true', help='Run prediction demo')
    parser.add_argument('--summary', action='store_true', help='Show project summary')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_prediction()
    elif args.summary:
        show_project_summary()
    else:
        print("Usage: python demo.py --demo or --summary")
        show_project_summary()