"""
Soft-Voting Ensemble Classifier with Probability Calibration for ShieldML.
Combines Random Forest, XGBoost, and LightGBM, and calibrates output probabilities
via CalibratedClassifierCV using the validation split.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import mlflow
from src.config import settings

class SoftVotingEnsemble(VotingClassifier):
    """
    Weighted Soft-Voting Ensemble over Random Forest, XGBoost, and LightGBM.
    Inherits from VotingClassifier for complete scikit-learn pipeline and tag compatibility.
    """
    _estimator_type = "classifier"

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        estimators = [
            ("rf", models[0]),
            ("xgb", models[1]),
            ("lgb", models[2])
        ]
        if weights is not None:
            w_sum = sum(weights)
            norm_weights = [w / w_sum for w in weights]
        else:
            norm_weights = [1.0 / len(models)] * len(models)
            
        super().__init__(estimators=estimators, voting="soft", weights=norm_weights)
        self.models = models
        self.estimators_ = list(models)
        self.named_estimators_ = {name: m for name, m in zip(["rf", "xgb", "lgb"], models)}
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

def build_calibrated_ensemble(
    rf_model: Any,
    xgb_model: Any,
    lgb_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    experiment_id: str,
    weights: List[float] = [0.25, 0.40, 0.35]
) -> Tuple[CalibratedClassifierCV, SoftVotingEnsemble, Dict[str, Any]]:
    """
    Build soft-voting ensemble and calibrate probabilities on the validation set.
    Logs run and calibration metrics (Brier score) to MLflow.
    """
    print(">>> [Ensemble] Constructing Soft-Voting Ensemble & Calibrating Probabilities...")
    
    raw_ensemble = SoftVotingEnsemble(
        models=[rf_model, xgb_model, lgb_model],
        weights=weights
    )
    
    # Calculate uncalibrated validation metrics
    uncal_probs = raw_ensemble.predict_proba(X_val)[:, 1]
    uncal_brier = float(brier_score_loss(y_val, uncal_probs))
    uncal_auc = float(roc_auc_score(y_val, uncal_probs))
    
    # Fit Platt scaling / sigmoid calibration on validation set
    if FrozenEstimator is not None:
        calibrated_clf = CalibratedClassifierCV(
            estimator=FrozenEstimator(raw_ensemble),
            method="sigmoid"
        )
    else:
        calibrated_clf = CalibratedClassifierCV(
            estimator=raw_ensemble,
            method="sigmoid",
            cv="prefit"
        )
    calibrated_clf.fit(X_val, y_val)
    
    cal_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    cal_brier = float(brier_score_loss(y_val, cal_probs))
    cal_auc = float(roc_auc_score(y_val, cal_probs))
    cal_pr = float(average_precision_score(y_val, cal_probs))
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="Calibrated-Soft-Voting-Ensemble") as run:
        mlflow.log_params({
            "ensemble_weights": str(weights),
            "calibration_method": "sigmoid (Platt Scaling)",
            "num_base_models": 3
        })
        mlflow.log_metrics({
            "val_uncalibrated_brier": uncal_brier,
            "val_calibrated_brier": cal_brier,
            "val_auc_roc": cal_auc,
            "val_auc_pr": cal_pr
        })
        run_id = run.info.run_id
        
    metrics = {
        "model": "CalibratedEnsemble",
        "weights": weights,
        "val_auc_roc": cal_auc,
        "val_auc_pr": cal_pr,
        "uncal_brier": uncal_brier,
        "cal_brier": cal_brier,
        "run_id": run_id
    }
    
    print(f"Ensemble Val AUC-ROC: {cal_auc:.4f}, AUC-PR: {cal_pr:.4f}")
    print(f"Calibration Brier Score: {uncal_brier:.4f} -> {cal_brier:.4f} (improved probability fidelity)")
    return calibrated_clf, raw_ensemble, metrics
