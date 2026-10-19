"""
SHAP TreeExplainer Explainability Engine for ShieldML.
Produces:
1. Global feature importance bar chart & beeswarm summary plot.
2. 3 local waterfall explanation plots for diverse transaction examples.
3. Sub-10ms local prediction explanations included in the live FastAPI /predict JSON response.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from src.config import settings

class ShieldMLExplainer:
    def __init__(self, ensemble_clf: Any, feature_names: List[str]):
        """
        Initializes TreeExplainer for the tree models in the ensemble.
        Ensemble TreeExplainer uses weighted combination of TreeExplainer instances.
        """
        self.ensemble = ensemble_clf
        self.feature_names = feature_names
        
        # Base models from ensemble
        raw_ens = ensemble_clf.estimator if hasattr(ensemble_clf, "estimator") else ensemble_clf
        if hasattr(raw_ens, "estimator"):
            raw_ens = raw_ens.estimator
        self.models = getattr(raw_ens, "models", [raw_ens])
        self.weights = getattr(raw_ens, "weights", [1.0 / len(self.models)] * len(self.models))
        
        print(f">>> [Explainability] Initializing SHAP TreeExplainers for {len(self.models)} tree models...")
        self.explainers = []
        for m in self.models:
            # TreeExplainer is exact and fast for tree models
            exp = shap.TreeExplainer(m)
            self.explainers.append(exp)
            
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted SHAP values across ensemble models."""
        total_shap = None
        for exp, weight in zip(self.explainers, self.weights):
            # shap_values shape can be (N, M, 2) or (N, M) depending on model type
            sv = exp.shap_values(X)
            if isinstance(sv, list):
                # Class 1 (fraud)
                sv = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
                
            if total_shap is None:
                total_shap = np.zeros_like(sv, dtype=float)
            total_shap += weight * sv
            
        return total_shap

    def generate_global_charts(self, X_sample: pd.DataFrame, experiment_id: str):
        """Generate and save global feature importance bar chart and summary plot."""
        print(">>> [Explainability] Generating Global SHAP Feature Importance Plots...")
        shap_vals = self.compute_shap_values(X_sample)
        
        # 1. Global Bar Chart
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            X_sample,
            plot_type="bar",
            feature_names=self.feature_names,
            show=False,
            color="#2563EB"
        )
        plt.title("Global Feature Importance (SHAP TreeExplainer) — ShieldML", fontsize=13, fontweight="bold")
        bar_path = settings.FIGURES_DIR / "shap_global_feature_importance.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # 2. Beeswarm Summary Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Feature Impact Summary Plot (Beeswarm)", fontsize=13, fontweight="bold")
        summary_path = settings.FIGURES_DIR / "shap_summary_plot.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="SHAP-Global-Explainability"):
            mlflow.log_artifact(str(bar_path))
            mlflow.log_artifact(str(summary_path))
            
        print(f"Saved global charts to {bar_path} and {summary_path}")

    def generate_waterfall_examples(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        probs: np.ndarray,
        experiment_id: str
    ):
        """
        Generate 3 distinct local waterfall plots:
        1. High-confidence fraud
        2. Clean legitimate transaction
        3. Borderline suspicious case
        """
        print(">>> [Explainability] Generating 3 Local Waterfall Plots for representative cases...")
        
        # Find representative indices
        fraud_mask = (y_test.values == 1)
        legit_mask = (y_test.values == 0)
        
        idx_fraud = np.where(fraud_mask & (probs > 0.85))[0]
        idx_fraud = idx_fraud[0] if len(idx_fraud) > 0 else np.argmax(probs)
        
        idx_legit = np.where(legit_mask & (probs < 0.05))[0]
        idx_legit = idx_legit[0] if len(idx_legit) > 0 else np.argmin(probs)
        
        idx_border = np.where(np.abs(probs - 0.50) < 0.15)[0]
        idx_border = idx_border[0] if len(idx_border) > 0 else int(len(probs) / 2)
        
        cases = [
            ("shap_waterfall_example_1_fraud.png", idx_fraud, f"Example 1: High-Risk Fraud (Score={probs[idx_fraud]:.3f})"),
            ("shap_waterfall_example_2_legit.png", idx_legit, f"Example 2: Clean Legitimate Tx (Score={probs[idx_legit]:.3f})"),
            ("shap_waterfall_example_3_borderline.png", idx_border, f"Example 3: Borderline Case (Score={probs[idx_border]:.3f})")
        ]
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="SHAP-Local-Waterfall-Examples"):
            for fname, idx, title in cases:
                sample_row = X_test.iloc[[idx]]
                row_shap = self.compute_shap_values(sample_row)[0]
                
                # Base value approximation
                base_val = float(np.mean(probs))
                
                exp_obj = shap.Explanation(
                    values=row_shap,
                    base_values=base_val,
                    data=sample_row.values[0],
                    feature_names=self.feature_names
                )
                
                plt.figure(figsize=(9, 5))
                shap.plots.waterfall(exp_obj, max_display=10, show=False)
                plt.title(title, fontsize=12, fontweight="bold")
                out_path = settings.FIGURES_DIR / fname
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(out_path))
                print(f"Saved {fname} (case: {title})")

    def explain_single(self, row_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fast runtime explanation for single-transaction serving (<5ms).
        Returns top features driving the score up or down.
        """
        if len(self.explainers) > 2:
            sv = self.explainers[2].shap_values(row_df)
        elif len(self.explainers) > 1:
            sv = self.explainers[1].shap_values(row_df)
        else:
            sv = self.explainers[0].shap_values(row_df)
            
        if isinstance(sv, list):
            shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            shap_vals = sv[0, :, 1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 2:
            shap_vals = sv[0]
        else:
            shap_vals = sv
            
        features = row_df.columns
        values = row_df.values[0]
        
        contributions = []
        for feat, val, s in zip(features, values, shap_vals):
            contributions.append({
                "feature": str(feat),
                "feature_value": round(float(val), 4) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": round(float(s), 4),
                "direction": "RISK_INCREASING" if s > 0 else "RISK_DECREASING"
            })
            
        # Sort by absolute SHAP magnitude
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:top_k]
