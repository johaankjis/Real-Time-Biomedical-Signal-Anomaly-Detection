# Biomedical Signal Anomaly Detection

Real-time anomaly detection system for biomedical signals using machine learning.

## Features

- **Synthetic Data Generation**: Simulates biomedical signals with injected anomalies (spikes, drift)
- **Feature Engineering**: Extracts time-series features (rolling stats, entropy, peaks, spectral features)
- **ML Models**: Isolation Forest + LSTM Autoencoder for anomaly detection
- **REST API**: FastAPI endpoint for real-time inference (<100ms)
- **Database Integration**: SQL/PL/SQL compatible persistence
- **Visualization**: ROC/PR curves, confusion matrix, feature importance
- **Experiment Tracking**: MLflow for hyperparameter tuning and model versioning

## Project Structure

\`\`\`
biomed-anomaly/
├── data/                    # Generated datasets
├── src/
│   ├── simulate_signals.py # Signal generation
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model training & evaluation
│   ├── serve.py            # FastAPI serving
│   └── db.py               # Database helpers
├── scripts/                # Executable scripts
├── models/                 # Saved models
├── requirements.txt
└── README.md
\`\`\`

## Quick Start

### 1. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Generate Synthetic Data

\`\`\`bash
python src/simulate_signals.py
\`\`\`

### 3. Extract Features

\`\`\`bash
python src/features.py
\`\`\`

### 4. Train Models

\`\`\`bash
python src/models.py
\`\`\`

### 5. Start API Server

\`\`\`bash
uvicorn src.serve:app --reload
\`\`\`

### 6. Test Inference

\`\`\`bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"signal": 1.5}]}'
\`\`\`

## Acceptance Criteria

- ✅ F1 Score ≥ 0.80 on synthetic anomalies
- ✅ REST endpoint < 100ms per request (local)
- ✅ Plots: ROC/PR, confusion matrix, feature importance
- ✅ MLflow logs: params, metrics, artifacts

## Tech Stack

- **Python**: NumPy, pandas, scikit-learn, TensorFlow
- **API**: FastAPI, Uvicorn
- **Database**: SQL/PL/SQL (Oracle-compatible)
- **ML Ops**: MLflow, SHAP
- **Visualization**: Matplotlib, Seaborn
