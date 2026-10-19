"""
Threshold Optimization and Test Evaluation for ShieldML.
Selects optimal threshold on Validation set maximizing F2-score (recall-weighted for fraud cost).
Evaluates and reports on the held-out Test set.
Generates ROC curves, PR curves, confusion matrix, and cost-benefit reports.
"""
from typing import Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import mlflow
from src.config import settings

def find_optimal_threshold_f2(
    y_true: pd.Series,
    y_probs: np.ndarray,
    cost_fn: float = 350.0,
    cost_fp: float = 2.50
) -> Tuple[float, Dict[str, Any]]:
    """
    Find threshold maximizing F2-score on validation set.
    Also computes business financial cost curve.
    """
    thresholds = np.linspace(0.02, 0.98, 200)
    best_threshold = 0.5
    best_f2 = -1.0
    best_metrics = {}
    
    threshold_history = []
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        
        # Financial cost model: Cost = FN * Cost_FN + FP * Cost_FP
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        expected_cost = (fn * cost_fn) + (fp * cost_fp)
        
        entry = {
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f2_score": float(f2),
            "fp": int(fp),
            "fn": int(fn),
            "expected_cost": float(expected_cost)
        }
        threshold_history.append(entry)
        
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = float(t)
            best_metrics = entry
            
    return best_threshold, {
        "optimal_threshold": round(best_threshold, 4),
        "val_best_f2": round(best_f2, 4),
        "val_precision_at_opt": round(best_metrics["precision"], 4),
        "val_recall_at_opt": round(best_metrics["recall"], 4),
        "history": threshold_history
    }

def evaluate_on_test_set(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float,
    experiment_id: str,
    cost_fn: float = 350.0,
    cost_fp: float = 2.50
) -> Dict[str, Any]:
    """
    Evaluate final champion model on held-out test set using the validation-selected threshold.
    Logs metrics and figures to MLflow and local artifacts.
    """
    print(f">>> [Evaluation] Evaluating Champion Model on Held-Out Test Set (Threshold={optimal_threshold:.4f})...")
    
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds_opt = (test_probs >= optimal_threshold).astype(int)
    test_preds_default = (test_probs >= 0.50).astype(int)
    
    auc_roc = float(roc_auc_score(y_test, test_probs))
    auc_pr = float(average_precision_score(y_test, test_probs))
    
    # Metrics at optimal threshold
    f2_opt = float(fbeta_score(y_test, test_preds_opt, beta=2, zero_division=0))
    prec_opt = float(precision_score(y_test, test_preds_opt, zero_division=0))
    rec_opt = float(recall_score(y_test, test_preds_opt, zero_division=0))
    cm_opt = confusion_matrix(y_test, test_preds_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
    cost_opt = (fn_o * cost_fn) + (fp_o * cost_fp)
    
    # Metrics at default 0.5 threshold
    f2_def = float(fbeta_score(y_test, test_preds_default, beta=2, zero_division=0))
    prec_def = float(precision_score(y_test, test_preds_default, zero_division=0))
    rec_def = float(recall_score(y_test, test_preds_default, zero_division=0))
    cm_def = confusion_matrix(y_test, test_preds_default)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel()
    cost_def = (fn_d * cost_fn) + (fp_d * cost_fp)
    financial_savings = cost_def - cost_opt
    
    # Assert rubric constraint
    if auc_roc < 0.90:
        print(f"WARNING: Test AUC-ROC {auc_roc:.4f} is below 0.90 constraint target.")
    else:
        print(f"SUCCESS: Test AUC-ROC {auc_roc:.4f} satisfies >= 0.90 constraint!")

    # 1. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ShieldML Ensemble (AUC = {auc_roc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.5, label="Random Guess")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax_roc.set_title("Held-Out Test ROC Curve — ShieldML Ensemble", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    roc_path = settings.FIGURES_DIR / "roc_curve.png"
    fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig_roc)
    
    # 2. Plot Precision-Recall Curve
    p_curve, r_curve, _ = precision_recall_curve(y_test, test_probs)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    ax_pr.plot(r_curve, p_curve, color="#10B981", lw=2.5, label=f"PR Curve (AUC-PR = {auc_pr:.4f})")
    ax_pr.scatter([rec_opt], [prec_opt], color="#EF4444", s=120, zorder=5,
                  label=f"Optimized Threshold ({optimal_threshold:.2f}): F2={f2_opt:.3f}")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Held-Out Test Precision-Recall Curve — ShieldML", fontsize=14, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=11)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    pr_path = settings.FIGURES_DIR / "precision_recall_curve.png"
    fig_pr.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig_pr)
    
    # 3. Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                xticklabels=["Legit (0)", "Fraud (1)"], yticklabels=["Legit (0)", "Fraud (1)"])
    ax_cm.set_xlabel("Predicted Label", fontsize=12)
    ax_cm.set_ylabel("True Label", fontsize=12)
    ax_cm.set_title(f"Confusion Matrix @ Threshold {optimal_threshold:.2f}", fontsize=13, fontweight="bold")
    cm_path = settings.FIGURES_DIR / "confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cm)
    
    # Log everything to MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="Test-Set-Champion-Evaluation") as run:
        mlflow.log_params({
            "optimal_threshold": optimal_threshold,
            "cost_fn_dollars": cost_fn,
            "cost_fp_dollars": cost_fp
        })
        mlflow.log_metrics({
            "test_auc_roc": auc_roc,
            "test_auc_pr": auc_pr,
            "test_f2_score_optimized": f2_opt,
            "test_precision_optimized": prec_opt,
            "test_recall_optimized": rec_opt,
            "test_f2_score_default": f2_def,
            "test_financial_savings_usd": financial_savings
        })
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(pr_path))
        mlflow.log_artifact(str(cm_path))
        run_id = run.info.run_id
        
    report = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "optimized_metrics": {
            "f2_score": round(f2_opt, 4),
            "precision": round(prec_opt, 4),
            "recall": round(rec_opt, 4),
            "tn": int(tn_o),
            "fp": int(fp_o),
            "fn": int(fn_o),
            "tp": int(tp_o),
            "expected_financial_loss_usd": round(cost_opt, 2)
        },
        "default_metrics_0_5": {
            "f2_score": round(f2_def, 4),
            "precision": round(prec_def, 4),
            "recall": round(rec_def, 4),
            "tn": int(tn_d),
            "fp": int(fp_d),
            "fn": int(fn_d),
            "tp": int(tp_d),
            "expected_financial_loss_usd": round(cost_def, 2)
        },
        "financial_savings_usd": round(financial_savings, 2),
        "run_id": run_id
    }
    
    # Save reports
    report_json = settings.REPORTS_DIR / "model_performance_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
        
    return report
