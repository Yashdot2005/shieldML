# ShieldML — Model Card

## 1. Model Details
- **Model Name:** ShieldML Financial Fraud Detection Champion Ensemble
- **Model Version:** 1.0.0
- **Model Type:** Soft-Voting Ensemble (Random Forest + XGBoost + LightGBM) with Sigmoid Probability Calibration (`CalibratedClassifierCV`)
- **Developers:** Code A Nova ML Engineering Sprint (Internship Program 2026)
- **Release Date:** October 2026
- **License:** MIT License
- **Frameworks:** scikit-learn 1.4+, XGBoost 2.0+, LightGBM 4.0+, SHAP 0.45+, FastAPI, MLflow 2.12+

---

## 2. Intended Use
- **Primary Use Case:** Real-time risk scoring and fraud classification of financial transaction events (card-present, card-not-present, contactless, mobile).
- **Serving Environment:** Low-latency REST API (`POST /predict`) evaluated with sub-50ms latency SLAs.
- **Decision Threshold:** Calibrated score evaluated against an $F_2$-optimized threshold selected on validation data to prioritize recall of fraudulent activity.
- **Out-of-Scope Applications:**
  - Long-term credit scoring or creditworthiness underwriting.
  - Multi-hop Anti-Money Laundering (AML) graph ring detection across multi-year accounts.
  - User biometric identity verification.

---

## 3. Training Data & Feature Engineering
- **Dataset:** Time-series financial transactions encompassing 60,000 records across 2,500 active accounts, 150 merchants, and 4,000 devices.
- **Class Imbalance:** Extreme imbalance with baseline fraud rate of ~1.2%.
- **Temporal Splitting:** Strict chronological split:
  - **Train Split (70%):** Monotonically sorted past transaction records used for parameter fitting and SMOTE oversampling.
  - **Validation Split (15%):** Intermediate chronological interval used strictly for Optuna hyperparameter optimization, probability calibration, and F2 threshold selection.
  - **Held-Out Test Split (15%):** Unseen future transactions used exclusively for final unbiased reporting.
- **Feature Set:**
  1. **Velocity Windows:** 1h, 24h, and 7-day backward rolling transaction counts and transaction amounts per card (guaranteeing zero future temporal leakage).
  2. **Merchant Amount Z-Score:** Difference between transaction amount and merchant's historical spend mean, scaled by merchant standard deviation (learned on train split only).
  3. **Cyclical Temporal Encoding:** Sinusoidal transformations of hour-of-day ($\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$) and day-of-week.
  4. **Device Fingerprint Profiling:** Historical device transaction frequencies and distinct cards per device.
  5. **Risk Domain Indicators:** Categorical encoding of merchant categories and non-domestic cross-border indicators.

---

## 4. Performance & Evaluation Metrics
Evaluated on the held-out test split (zero leakage):

| Metric | Target Constraint | Achieved Champion Result |
| :--- | :--- | :--- |
| **AUC-ROC** | **> 0.9000** | **> 0.9500 (PASSED)** |
| **AUC-PR** | High precision at recall | **> 0.8200** |
| **F2-Score** | Optimized on Validation | **> 0.8500** |
| **Recall (Sensitivity)** | Prioritized over false alarms | **> 88.0%** |
| **Prediction Latency** | **< 50ms (P95)** | **~5 - 15ms (PASSED)** |

### Business Cost Rationale ($F_2$ Maximization)
In financial payments:
- **Cost of a False Negative (Missed Fraud):** $\approx \$350$ (direct chargeback liability, merchant settlement fines, card reissuance).
- **Cost of a False Positive (False Alarm):** $\approx \$2.50$ (automated 2FA challenge SMS, transient user friction).
- Therefore, minimizing false negatives is over $100\times$ more critical than false positives. The $F_2$ metric weights recall twice as heavily as precision ($\beta = 2$), achieving significant net financial loss reduction compared to the arbitrary default 0.50 cutoff.

---

## 5. Explainability & Compliance
- **Methodology:** `shap.TreeExplainer` applied directly to tree ensemble constituents.
- **Local Predictions:** Every `/predict` API response delivers the top 5 risk-increasing and risk-decreasing features with exact SHAP attribution and feature values.
- **Global Governance:** Feature importance plots and 3 representative waterfall profiles (clear fraud, clear legitimate, and borderline) are archived in MLflow and exported to `artifacts/figures/`.

---

## 6. Known Limitations & Failure Modes
1. **Cold Start Accounts:** Newly issued cards or newly created digital wallets lack 7-day velocity history. Fallbacks to global merchant baselines and category risk priors are used until 3 transactions are logged.
2. **Flash Mob / Coordinated Swarm Fraud:** Sudden bot-driven attacks using rotating residential proxies and synthetic identities may bypass single-card velocity checks.
3. **Concept Drift:** Macroeconomic shifts (holiday shopping surges, promotional discount seasons) alter normal ticket distributions. Requires automated weekly drift detection and model retraining.
4. **Sub-Optimal Network Topology:** If the serving API is separated from the feature store by large network hops (>30ms), the sub-50ms SLA could degrade. ShieldML utilizes an in-memory sliding buffer to guarantee sub-millisecond feature resolution.
