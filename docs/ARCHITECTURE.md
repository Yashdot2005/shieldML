# ShieldML Architecture & Data Flow

This document outlines the end-to-end system architecture of **ShieldML**, detailing data ingestion, feature engineering, offline training, live serving, and monitoring.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    subgraph Data & Pipeline Layer
        A[Raw Transactions CSV / Stream] --> B[Schema Validation & Chronological Sort]
        B --> C[Time-Based Stratified Split<br/>Train: 70% | Val: 15% | Test: 15%]
        C --> D[Feature Engineering Engine<br/>Fitted on Train Only]
        D --> E[SMOTE Oversampling<br/>Applied to Train Only]
    end

    subgraph Training & MLOps Layer
        E --> F1[Random Forest + GridSearchCV]
        D --> F2[XGBoost + Optuna Tuning]
        D --> F3[LightGBM + Optuna Tuning]
        F1 & F2 & F3 --> G[Soft-Voting Ensemble]
        G --> H[CalibratedClassifierCV<br/>Platt Sigmoid on Val Set]
        H --> I[F2-Score Threshold Optimization<br/>Maximized on Val Split]
        I --> J[SHAP TreeExplainer Artifacts]
        H --> K[MLflow Tracking & Model Registry<br/>Champion Staging]
    end

    subgraph Real-Time Serving Layer
        L[Client Application / Payment Gateway] -->|POST /predict <50ms| M[FastAPI Serving Application]
        M --> N[Lightweight In-Memory Feature Store<br/>1h, 24h, 7d Velocity in <1ms]
        M --> O[Loaded Champion Ensemble in Memory]
        O --> P[Calibrated Fraud Score & F2 Decision]
        O --> Q[Real-Time SHAP Local Attribution]
        P & Q -->|JSON Response| L
    end

    subgraph Observability & Monitoring
        M --> R[Rolling Metrics Tracker<br/>Last 100 Window]
        R --> S[GET /metrics Endpoint]
        R --> T[Interactive Dashboard UI]
    end
```

---

## 2. Core Architectural Principles

### 2.1 Zero Temporal Data Leakage Guarantee
In transactional time series, computing aggregate statistics (e.g., standard deviation of merchant spend or historical transaction frequencies) over the entire dataset introduces **lookahead bias**. ShieldML strictly prevents this:
1. **Chronological Splitting:** Transactions are strictly ordered by timestamp. Training constitutes the past, Validation constitutes intermediate transactions, and Test constitutes future unseen records.
2. **Train-Only Transformation:** `ShieldMLFeatureEngineer.fit()` operates solely on the training partition. Unseen entities in Validation or Test fall back to global training priors.
3. **Past-Only Rolling Windows:** Velocity metrics for 1h, 24h, and 7-day windows are calculated with left-closed bounds (`closed='left'`), ensuring that the current transaction is never included in its own historical velocity.

### 2.2 Class Imbalance Handling
Financial fraud exhibits natural class imbalance (~1.2%). Resampling validation or test sets leads to artificially inflated metrics that fail in production.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Applied exclusively to the training set (`X_train, y_train`), lifting minority representation to ~8% to prevent decision boundary starvation.
- **Cost-Sensitive Loss:** Tree models incorporate `scale_pos_weight` ($N_{\text{negative}} / N_{\text{positive}}$) to penalize missed fraud cases.
- **Validation & Test Integrity:** Validation and test splits remain un-sampled and pristine.

### 2.3 Ensemble & Probability Calibration
Individual classifiers exhibit idiosyncratic variance. ShieldML implements a soft-voting ensemble:
$$\hat{p}_{\text{ensemble}}(\mathbf{x}) = w_{\text{RF}} p_{\text{RF}}(\mathbf{x}) + w_{\text{XGB}} p_{\text{XGB}}(\mathbf{x}) + w_{\text{LGB}} p_{\text{LGB}}(\mathbf{x})$$

Because raw tree ensemble outputs are frequently overconfident or distorted near 0 and 1, `CalibratedClassifierCV` (Platt Sigmoid Scaling) is fitted on the validation set. This ensures that a predicted score of `0.85` corresponds to an empirical 85% probability of fraud.

### 2.4 $F_2$-Score Threshold Optimization
The conventional 0.50 cutoff fails in asymmetric payment loss environments.
- **False Negative Cost ($C_{\text{FN}}$):** $\approx \$350$
- **False Positive Cost ($C_{\text{FP}}$):** $\approx \$2.50$
$$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{P \cdot R}{4P + R}$$
The optimal decision threshold $t^*$ is discovered strictly on the validation set by maximizing $F_2$, and evaluated on the held-out test split.

### 2.5 Low-Latency Serving Pipeline (< 50ms SLA)
To satisfy stringent payment gateway SLAs (< 50ms):
1. **Model & Explainer Pre-loaded:** Models and SHAP explainers are held in RAM via FastAPI's `lifespan` handler.
2. **In-Memory Feature Store:** An in-memory sliding ring buffer maintains 7-day transaction history per card, answering 1h/24h/7d velocity queries in `< 0.5ms`.
3. **Vectorized TreeExplainer:** SHAP tree traversal completes in `< 5ms`, delivering exact feature attributions inside the prediction response payload.
