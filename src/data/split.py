"""
Time-based Train/Validation/Test Split for ShieldML.
Prevents future temporal leakage by partitioning chronologically.
"""
from typing import Tuple
import pandas as pd
from src.config import settings
from src.data.schema import validate_dataframe_schema

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transactional data chronologically into Train, Validation, and Test sets.
    Ensures strict temporal ordering to prevent data leakage.
    """
    validate_dataframe_schema(df)
    
    # Ensure datetime sorting
    df = df.copy()
    df["_dt"] = pd.to_datetime(df[settings.TIME_COL])
    df = df.sort_values("_dt").reset_index(drop=True)
    df = df.drop(columns=["_dt"])
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """Save split datasets to processed directory as Parquet files."""
    proc_dir = settings.PROCESSED_DATA_DIR
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "train": proc_dir / "train.parquet",
        "val": proc_dir / "val.parquet",
        "test": proc_dir / "test.parquet"
    }
    
    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)
    
    stats = {
        "train": {
            "count": len(train_df),
            "fraud_count": int(train_df[settings.TARGET_COL].sum()),
            "fraud_rate": float(train_df[settings.TARGET_COL].mean()),
            "time_range": (train_df[settings.TIME_COL].min(), train_df[settings.TIME_COL].max())
        },
        "val": {
            "count": len(val_df),
            "fraud_count": int(val_df[settings.TARGET_COL].sum()),
            "fraud_rate": float(val_df[settings.TARGET_COL].mean()),
            "time_range": (val_df[settings.TIME_COL].min(), val_df[settings.TIME_COL].max())
        },
        "test": {
            "count": len(test_df),
            "fraud_count": int(test_df[settings.TARGET_COL].sum()),
            "fraud_rate": float(test_df[settings.TARGET_COL].mean()),
            "time_range": (test_df[settings.TIME_COL].min(), test_df[settings.TIME_COL].max())
        }
    }
    
    return stats
