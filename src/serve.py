"""
FastAPI serving endpoint for real-time anomaly detection.
Provides REST API for inference with <100ms latency.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import pandas as pd
import numpy as np
import time
import os
from features import featurize

# Initialize FastAPI app
app = FastAPI(
    title="Biomedical Signal Anomaly Detection API",
    description="Real-time anomaly detection for biomedical signals",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE = {}


def load_models():
    """Load trained models into memory."""
    global MODEL_CACHE
    
    if not MODEL_CACHE:
        print("Loading models...")
        
        model_path = "models/iforest.joblib"
        scaler_path = "models/scaler.joblib"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first."
            )
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. Please train the model first."
            )
        
        MODEL_CACHE['iforest'] = joblib.load(model_path)
        MODEL_CACHE['scaler'] = joblib.load(scaler_path)
        
        # Load threshold from metrics
        metrics_path = "models/metrics.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            MODEL_CACHE['threshold'] = metrics_df['threshold'].values[0]
        else:
            MODEL_CACHE['threshold'] = None
        
        print("Models loaded successfully!")
    
    return MODEL_CACHE


# Pydantic models for request/response
class SignalPoint(BaseModel):
    """Single signal data point."""
    ts: float = Field(..., description="Timestamp in seconds")
    signal: float = Field(..., description="Signal value")


class PredictRequest(BaseModel):
    """Request body for prediction endpoint."""
    signals: List[SignalPoint] = Field(
        ..., 
        description="List of signal data points",
        min_items=1
    )
    freq: Optional[int] = Field(
        100, 
        description="Sampling frequency in Hz"
    )
    window: Optional[int] = Field(
        50, 
        description="Window size for feature extraction"
    )


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""
    anomaly: bool = Field(..., description="Whether the signal is anomalous")
    score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    threshold: Optional[float] = Field(None, description="Threshold used for classification")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    num_samples: int = Field(..., description="Number of samples processed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    rows: List[Dict[str, float]] = Field(
        ...,
        description="List of signal data points as dictionaries"
    )


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Could not load models on startup: {e}")
        print("Models will be loaded on first request.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Biomedical Signal Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        models = load_models()
        model_loaded = 'iforest' in models and 'scaler' in models
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict anomalies in biomedical signal data.
    
    Returns anomaly classification and score for the most recent signal point.
    """
    start_time = time.time()
    
    try:
        # Load models
        models = load_models()
        iforest = models['iforest']
        scaler = models['scaler']
        threshold = models.get('threshold')
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {"ts": point.ts, "signal": point.signal}
            for point in request.signals
        ])
        
        # Extract features
        X = featurize(df, window=request.window, freq=request.freq)
        
        # Standardize
        X_scaled = scaler.transform(X)
        
        # Get anomaly score for the last point
        scores = -iforest.decision_function(X_scaled)
        last_score = float(scores[-1])
        
        # Determine if anomalous
        if threshold is not None:
            is_anomaly = last_score > threshold
        else:
            # Fallback: use 99th percentile of current batch
            threshold = float(np.percentile(scores, 99))
            is_anomaly = last_score > threshold
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictResponse(
            anomaly=bool(is_anomaly),
            score=last_score,
            threshold=threshold,
            latency_ms=latency_ms,
            num_samples=len(request.signals)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest):
    """
    Batch prediction endpoint (legacy format).
    
    Accepts a list of signal dictionaries and returns anomaly predictions.
    """
    start_time = time.time()
    
    try:
        # Load models
        models = load_models()
        iforest = models['iforest']
        scaler = models['scaler']
        threshold = models.get('threshold')
        
        # Convert to DataFrame
        df = pd.DataFrame(request.rows)
        
        if 'signal' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Request must contain 'signal' column"
            )
        
        # Add timestamp if missing
        if 'ts' not in df.columns:
            df['ts'] = np.arange(len(df)) / 100.0  # Assume 100 Hz
        
        # Extract features
        X = featurize(df, window=50, freq=100)
        
        # Standardize
        X_scaled = scaler.transform(X)
        
        # Get anomaly scores
        scores = -iforest.decision_function(X_scaled)
        
        # Determine threshold
        if threshold is None:
            threshold = float(np.percentile(scores, 99))
        
        # Predict anomalies
        predictions = (scores > threshold).astype(int)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Return results
        results = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            results.append({
                "index": i,
                "anomaly": bool(pred),
                "score": float(score)
            })
        
        return {
            "predictions": results,
            "threshold": threshold,
            "latency_ms": latency_ms,
            "num_samples": len(request.rows)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Biomedical Signal Anomaly Detection API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
