"""
Configuration and settings for ShieldML Fraud Detection System.
"""
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ShieldML — Real-Time Financial Fraud Detection"
    VERSION: str = "1.0.0"
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "transactions.csv"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
    REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
    
    # MLflow (SQLite database backend supports model registry and latest tracking)
    MLFLOW_TRACKING_URI: str = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
    MLFLOW_EXPERIMENT_NAME: str = "ShieldML-Fraud-Detection"
    CHAMPION_MODEL_NAME: str = "ShieldML-Ensemble-Champion"
    
    # Data generation / splitting params
    RANDOM_SEED: int = 42
    TEST_SPLIT_RATIO: float = 0.15
    VAL_SPLIT_RATIO: float = 0.15
    
    # Target and identifier columns
    TARGET_COL: str = "is_fraud"
    TIME_COL: str = "timestamp"
    USER_COL: str = "user_id"
    CARD_COL: str = "card_id"
    AMOUNT_COL: str = "amount"
    MERCHANT_COL: str = "merchant_id"
    DEVICE_COL: str = "device_id"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LATENCY_TARGET_MS: float = 50.0

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
for path in [
    settings.DATA_DIR / "raw",
    settings.PROCESSED_DATA_DIR,
    settings.MODELS_DIR,
    settings.FIGURES_DIR,
    settings.REPORTS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
