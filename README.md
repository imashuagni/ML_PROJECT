# Purchase Value Prediction - ML Project

A machine learning project for predicting purchase values from web analytics data using various regression models.

## 📊 Project Overview

This project tackles a challenging regression problem where we predict customer purchase values based on comprehensive web analytics data. The dataset contains user behavior, device information, geographic data, and traffic source details from web sessions.

### Problem Statement
- **Task**: Regression (Predict continuous purchase values)
- **Target**: `purchaseValue` - monetary value of customer purchases
- **Challenge**: Highly imbalanced dataset with many zero-value purchases
- **Features**: 51+ features including device info, geo location, traffic sources, and user engagement metrics

## 🗂️ Project Structure

```
ML_PROJECT/
├── data/
│   ├── raw/                    # Original dataset files
│   │   ├── train_data.csv      # Training data (116K+ samples)
│   │   ├── test_data.csv       # Test data (29K+ samples)
│   │   └── sample_submission.csv
│   └── processed/              # Processed data files
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   └── model_training.py       # Model training and evaluation
├── notebooks/
│   └── data_exploration.ipynb  # Exploratory data analysis
├── models/                     # Trained model files
├── results/                    # Model outputs and evaluations
├── train_pipeline.py          # Main training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 📋 Dataset Details

- **Training Data**: 116,024 samples × 52 features
- **Test Data**: 29,007 samples × 51 features
- **Features Include**:
  - **Traffic Sources**: Campaign info, referral paths, keywords
  - **Device Information**: Browser, OS, mobile device details  
  - **Geographic Data**: Country, region, city, network location
  - **User Engagement**: Page views, session data, bounce rates
  - **Technical Details**: Screen resolution, browser version, etc.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/imashuagni/ML_PROJECT.git
cd ML_PROJECT

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Execute the full ML pipeline
python train_pipeline.py
```

This will:
- Load and preprocess the data
- Train multiple models (Linear Regression, Random Forest, XGBoost, LightGBM)
- Evaluate models on validation data
- Generate predictions and submission file
- Save the best model

### 3. Explore the Data

```bash
# Start Jupyter notebook for exploration
jupyter notebook notebooks/data_exploration.ipynb
```

## 🔧 Pipeline Components

### Data Processing (`src/data_processing.py`)
- **Missing Value Handling**: Median for numeric, mode for categorical
- **Categorical Encoding**: Label encoding for categorical features
- **Feature Scaling**: StandardScaler for numerical features
- **Train/Validation Split**: 80/20 split for model validation

### Model Training (`src/model_training.py`)
Supports multiple algorithms:
- **Linear Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting (if installed)
- **LightGBM**: Fast gradient boosting (if installed)

### Evaluation Metrics
- **RMSE**: Root Mean Square Error (primary metric)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## 📈 Results

After running the pipeline, you'll find:
- `results/model_evaluation.csv`: Comparison of all models
- `results/submission.csv`: Final predictions for submission
- `results/{model}_feature_importance.csv`: Feature importance rankings
- `models/best_model_{name}.pkl`: Saved best performing model

## 🛠️ Advanced Usage

### Custom Model Training

```python
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

# Initialize components
processor = DataProcessor()
trainer = ModelTrainer()

# Load and process data
train_df, test_df = processor.load_data('data/raw/train_data.csv', 'data/raw/test_data.csv')
X_train, X_test, y_train = processor.preprocess_data(train_df, test_df)

# Train specific model
trainer.train_model('random_forest', X_train, y_train)
predictions = trainer.predict('random_forest', X_test)
```

### Adding New Models

Extend the `ModelTrainer` class by adding new model configurations:

```python
trainer.model_configs['new_model'] = {
    'model_class': YourModelClass,
    'params': {'param1': value1, 'param2': value2}
}
```

## 📊 Key Features and Insights

- **Imbalanced Target**: ~90% of samples have zero purchase value
- **Rich Feature Set**: Web analytics provide comprehensive user behavior data
- **Geographic Diversity**: Global user base with regional patterns
- **Device Variety**: Multiple browsers, OS, and device types
- **Traffic Sources**: Organic, referral, social, and direct traffic

## 🔍 Model Performance Expectations

Given the nature of the problem:
- **Challenge**: Predicting exact purchase values is difficult due to sparsity
- **Success Metrics**: Focus on RMSE and ability to identify high-value customers
- **Baseline Performance**: Simple models may struggle with zero-inflation
- **Advanced Models**: Tree-based models typically perform better on this type of data

## 📝 Dependencies

Core requirements:
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `matplotlib`/`seaborn`: Data visualization
- `jupyter`: Interactive analysis

Optional (for better performance):
- `xgboost`: Gradient boosting
- `lightgbm`: Fast gradient boosting
- `catboost`: Categorical boosting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🏆 Competition Context

This appears to be based on a machine learning competition focused on converting web clicks to purchase value predictions. The goal is to help businesses understand which website interactions are most likely to lead to revenue.

---

*For questions or issues, please open a GitHub issue or contact the repository maintainer.*