"""
FastAPI High-Performance Serving Application for ShieldML.
Provides:
- POST /predict : Live fraud scoring, calibrated probability, and SHAP explainability in <50ms.
- GET  /metrics : Monitoring stub reporting count, rolling latency, and fraud rate over last 100.
- GET  /health  : Readiness and health status.
- UI   /        : Interactive visual fraud operations dashboard.
"""
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib

from src.config import settings
from src.api.schemas import TransactionPayload, PredictionResponse, MetricsResponse
from src.api.metrics_tracker import MetricsTracker
from src.features.feature_store import LightweightFeatureStore

# Global singletons loaded at startup
metrics_tracker = MetricsTracker(window_size=100)
feature_store = LightweightFeatureStore(ttl_days=7)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, transformers, and explainer once into memory at startup."""
    print(">>> [FastAPI Lifespan] Starting ShieldML API server...")
    model_path = settings.MODELS_DIR / "champion_ensemble.joblib"
    fe_path = settings.MODELS_DIR / "feature_engineer.joblib"
    exp_path = settings.MODELS_DIR / "explainer.joblib"
    meta_path = settings.MODELS_DIR / "metadata.json"
    
    if not (model_path.exists() and fe_path.exists()):
        print("WARNING: Model artifacts not found. Please run 'python run_pipeline.py' first.")
        app.state.model = None
        app.state.fe = None
        app.state.explainer = None
        app.state.metadata = {"optimal_threshold": 0.35, "model_name": "Uninitialized"}
    else:
        print(f"Loading champion model from {model_path}...")
        app.state.model = joblib.load(model_path)
        app.state.fe = joblib.load(fe_path)
        app.state.explainer = joblib.load(exp_path)
        
        if meta_path.exists():
            with open(meta_path, "r") as f:
                app.state.metadata = json.load(f)
        else:
            app.state.metadata = {"optimal_threshold": 0.35, "model_name": settings.CHAMPION_MODEL_NAME}

        # Optimize tree models for low-latency single-row inference (eliminates joblib thread-pool overhead)
        ens = app.state.model.estimator if hasattr(app.state.model, "estimator") else app.state.model
        if hasattr(ens, "estimator"):
            ens = ens.estimator
        for m in getattr(ens, "models", []):
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1

        print(f"SUCCESS: Champion Model loaded! Optimal Threshold: {app.state.metadata.get('optimal_threshold')}")

    yield
    print(">>> [FastAPI Lifespan] Shutting down ShieldML API server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade real-time fraud detection scoring service with SHAP explainability and sub-50ms latency.",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for interactive web dashboard
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Service health and model readiness check."""
    is_ready = getattr(app.state, "model", None) is not None
    meta = getattr(app.state, "metadata", {}) if hasattr(app.state, "metadata") else {}
    return {
        "status": "HEALTHY" if is_ready else "DEGRADED",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": is_ready,
        "model_name": meta.get("model_name", "None") if meta else "None",
        "threshold": meta.get("optimal_threshold", 0.35) if meta else 0.35
    }

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Monitoring Stub: returns prediction count, average latency, p95 latency,
    and fraud rate over the last 100 predictions.
    """
    return metrics_tracker.get_summary()

@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict_fraud(payload: TransactionPayload):
    """
    Score a single financial transaction in real-time.
    Loads champion model at startup; processes velocity and SHAP in <50ms.
    """
    t_start = time.perf_counter()
    
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded. Please run the training pipeline first."
        )

    tx_dict = payload.model_dump()
    card_id = tx_dict["card_id"]
    timestamp = tx_dict["timestamp"]
    
    # 1. Look up past velocity from lightweight feature store (<0.5ms)
    past_velocity = feature_store.get_card_velocity(card_id, timestamp)
    
    # 2. Transform into feature row (<1ms)
    feature_row = app.state.fe.transform_single(tx_dict, past_velocity)
    
    # 3. Model scoring (<2ms)
    probs = app.state.model.predict_proba(feature_row)
    fraud_score = float(probs[0, 1])
    
    # 4. Apply optimal validation threshold
    threshold = float(app.state.metadata.get("optimal_threshold", 0.35))
    is_fraud = fraud_score >= threshold
    decision = "FRAUD_ALERT" if is_fraud else "APPROVE"
    
    # 5. Generate SHAP explanation (<5ms)
    shap_explanation = app.state.explainer.explain_single(feature_row, top_k=5)
    
    # 6. Update Feature Store and Metrics Tracker (<0.2ms)
    feature_store.record_transaction(card_id, timestamp, tx_dict["amount"])
    
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    metrics_tracker.record_prediction(elapsed_ms, is_fraud)
    
    return PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_score=round(fraud_score, 4),
        is_fraud=is_fraud,
        decision=decision,
        threshold_applied=threshold,
        latency_ms=round(elapsed_ms, 2),
        model_version=app.state.metadata.get("model_name", "Champion-v1"),
        shap_explanation=shap_explanation
    )

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Serve the interactive web control center."""
    index_file = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>ShieldML API is running. Visit /docs for Swagger UI.</h2>")
