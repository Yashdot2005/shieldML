"""
XGBoost Classifier Training and Optuna Hyperparameter Tuning with MLflow Nested Child Runs.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, fbeta_score, average_precision_score
import optuna
import mlflow
from src.config import settings

# Suppress Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    experiment_id: str,
    n_trials: int = 12
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Train XGBoost with Optuna hyperparameter tuning, logging child runs to MLflow."""
    print(">>> [Training] Starting XGBoost Optuna Tuning (with MLflow child runs)...")
    
    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="XGBoost-Optuna-Parent") as parent_run:
        parent_run_id = parent_run.info.run_id
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 60, 160),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, scale_pos_weight),
                "random_state": settings.RANDOM_SEED,
                "eval_metric": "auc",
                "n_jobs": -1
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            val_probs = model.predict_proba(X_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_probs))
            
            # Log each trial as an MLflow child run
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"XGBoost-Trial-{trial.number}",
                nested=True
            ):
                mlflow.log_params(params)
                mlflow.log_metric("trial_val_auc_roc", val_auc)
                
            return val_auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params.update({
            "random_state": settings.RANDOM_SEED,
            "eval_metric": "auc",
            "n_jobs": -1
        })
        
        # Train final champion model on best parameters
        best_xgb = xgb.XGBClassifier(**best_params)
        best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_probs = best_xgb.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        auc_roc = float(roc_auc_score(y_val, val_probs))
        auc_pr = float(average_precision_score(y_val, val_probs))
        f2 = float(fbeta_score(y_val, val_preds, beta=2, zero_division=0))
        
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "val_auc_roc": auc_roc,
            "val_auc_pr": auc_pr,
            "val_f2_score": f2
        })
        
        metrics = {
            "model": "XGBoost",
            "best_params": best_params,
            "val_auc_roc": auc_roc,
            "val_auc_pr": auc_pr,
            "val_f2_score": f2,
            "run_id": parent_run_id
        }
        
        print(f"XGBoost Best AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, F2: {f2:.4f}")
        return best_xgb, metrics
