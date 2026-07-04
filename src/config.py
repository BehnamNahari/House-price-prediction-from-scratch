from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODEL_PATH = MODELS_DIR / "linear_regression.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

RANDOM_SEED = 42
VAL_RATIO = 0.2

TARGET_COLUMN = "SalePrice"

# baseline settings
LEARNING_RATE = 0.01
N_EPOCHS = 3000
LAMBDA_REG = 0.1