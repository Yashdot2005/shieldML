"""
Random Forest Classifier Training and GridSearchCV Tuning with MLflow Logging.
"""
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score, average_precision_score
import mlflow
from src.config import settings

def train_random_forest_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    experiment_id: str
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train Random Forest with GridSearchCV tuning and log run to MLflow."""
    print(">>> [Training] Starting Random Forest GridSearchCV...")
    
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [8, 14, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", "balanced_subsample"]
    }
    
    base_rf = RandomForestClassifier(random_state=settings.RANDOM_SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.RANDOM_SEED)
    
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="RandomForest-GridSearch") as run:
        grid.fit(X_train, y_train)
        best_rf: RandomForestClassifier = grid.best_estimator_
        
        # Validation metrics
        val_probs = best_rf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        auc_roc = float(roc_auc_score(y_val, val_probs))
        auc_pr = float(average_precision_score(y_val, val_probs))
        f2 = float(fbeta_score(y_val, val_preds, beta=2, zero_division=0))
        
        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "val_auc_roc": auc_roc,
            "val_auc_pr": auc_pr,
            "val_f2_score": f2,
            "cv_best_score": float(grid.best_score_)
        })
        
        metrics = {
            "model": "RandomForest",
            "best_params": grid.best_params_,
            "val_auc_roc": auc_roc,
            "val_auc_pr": auc_pr,
            "val_f2_score": f2,
            "run_id": run.info.run_id
        }
        
        print(f"Random Forest Best AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, F2: {f2:.4f}")
        return best_rf, metrics
