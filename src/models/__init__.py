"""Models package for ShieldML."""
from src.models.imbalance import apply_smote, benchmark_imbalance_strategies
from src.models.train_rf import train_random_forest_grid_search
from src.models.train_xgb import train_xgboost_optuna
from src.models.train_lgb import train_lightgbm_optuna
from src.models.ensemble import SoftVotingEnsemble, build_calibrated_ensemble
from src.models.threshold import find_optimal_threshold_f2, evaluate_on_test_set
from src.models.explain import ShieldMLExplainer

__all__ = [
    "apply_smote",
    "benchmark_imbalance_strategies",
    "train_random_forest_grid_search",
    "train_xgboost_optuna",
    "train_lightgbm_optuna",
    "SoftVotingEnsemble",
    "build_calibrated_ensemble",
    "find_optimal_threshold_f2",
    "evaluate_on_test_set",
    "ShieldMLExplainer"
]
