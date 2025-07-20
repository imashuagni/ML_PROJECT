# ML Pipeline Configuration File

# Data paths
TRAIN_DATA_PATH = "data/raw/train_data.csv"
TEST_DATA_PATH = "data/raw/test_data.csv"
SAMPLE_SUBMISSION_PATH = "data/raw/sample_submission.csv"

# Output paths
RESULTS_DIR = "results"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Model configuration
MODELS_TO_TRAIN = [
    "linear_regression",
    "random_forest"
    # Uncomment if you have XGBoost/LightGBM installed
    # "xgboost",
    # "lightgbm"
]

# Training configuration
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
}