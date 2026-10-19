"""
Unit tests for ShieldML Imbalance handling, Ensemble, Calibration, and Threshold Optimization.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.imbalance import apply_smote
from src.models.ensemble import SoftVotingEnsemble
from src.models.threshold import find_optimal_threshold_f2

def test_smote_oversampling_resamples_properly():
    """Verify SMOTE correctly increases the minority fraud class representation."""
    np.random.seed(42)
    # 990 legit, 10 fraud
    X = pd.DataFrame(np.random.randn(1000, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series([0] * 990 + [1] * 10, name="is_fraud")
    
    X_res, y_res = apply_smote(X, y, sampling_strategy=0.10)
    
    assert len(X_res) > len(X)
    assert y_res.sum() > 10
    # Minority ratio should now be ~10% of majority class
    assert y_res.sum() == int(990 * 0.10)

def test_soft_voting_ensemble_probabilities():
    """Verify soft-voting ensemble computes weighted average probabilities properly."""
    # Dummy models
    class MockModel:
        def __init__(self, prob):
            self.prob = prob
        def predict_proba(self, X):
            return np.array([[1.0 - self.prob, self.prob]])
            
    m1 = MockModel(0.20)
    m2 = MockModel(0.40)
    m3 = MockModel(0.60)
    
    ens = SoftVotingEnsemble(models=[m1, m2, m3], weights=[0.25, 0.50, 0.25])
    probs = ens.predict_proba(pd.DataFrame([[1, 2]]))
    
    expected = 0.25 * 0.20 + 0.50 * 0.40 + 0.25 * 0.60
    assert abs(probs[0, 1] - expected) < 1e-5

def test_optimal_f2_threshold_selection():
    """Verify threshold search selects threshold that maximizes F2 score."""
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted probabilities
    y_probs = np.array([0.1, 0.2, 0.35, 0.45, 0.38, 0.7, 0.85, 0.9])
    
    best_thresh, details = find_optimal_threshold_f2(y_true, y_probs)
    assert 0.0 < best_thresh < 1.0
    assert details["val_best_f2"] > 0.0
