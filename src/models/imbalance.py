"""
Class Imbalance Handling and Benchmarking for ShieldML.
Supports SMOTE, ADASYN, Random Under-Sampling, and Cost-Sensitive Weighting.
Applies resampling ONLY to the training set — never to validation or test splits.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import mlflow

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling strictly to training data.
    Increases minority fraud ratio up to sampling_strategy (e.g. 10%).
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

def apply_adasyn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply ADASYN adaptive synthetic oversampling strictly to training data."""
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply RandomUnderSampler strictly to training data."""
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

def calculate_cost_sensitive_weight(y_train: pd.Series) -> float:
    """Calculate scale_pos_weight: ratio of negative to positive class."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / max(1, n_pos))

def benchmark_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> pd.DataFrame:
    """
    Systematically compare imbalance mitigation strategies:
    1. Baseline (Raw Imbalance)
    2. Cost-Sensitive Class Weighting
    3. Random Under-Sampling
    4. SMOTE (Synthetic Minority Over-sampling)
    5. ADASYN (Adaptive Synthetic)
    """
    results = []
    
    strategies = {
        "Baseline": (X_train, y_train, None),
        "Cost-Sensitive": (X_train, y_train, "balanced"),
        "Random-Undersampling": (*apply_undersampling(X_train, y_train), None),
        "SMOTE": (*apply_smote(X_train, y_train), None),
        "ADASYN": (*apply_adasyn(X_train, y_train), None)
    }
    
    for name, (X_tr, y_tr, class_weight) in strategies.items():
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        auc = roc_auc_score(y_val, probs)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        
        results.append({
            "Strategy": name,
            "AUC_ROC": round(float(auc), 4),
            "F2_Score_0.5": round(float(f2), 4),
            "Precision_0.5": round(float(prec), 4),
            "Recall_0.5": round(float(rec), 4),
            "Train_Samples": len(X_tr),
            "Train_Fraud_Count": int(y_tr.sum())
        })
        
    df_results = pd.DataFrame(results)
    return df_results
