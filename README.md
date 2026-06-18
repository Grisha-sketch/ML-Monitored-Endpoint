# ML Monitored Endpoint

![CI](https://github.com/Grisha-sketch/ml-monitored-endpoint/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

A production-style ML inference API with a live monitoring dashboard, Prometheus metrics, and a full CI/CD pipeline.

## Features

- Random Forest classifier served via FastAPI REST API
- Live monitoring dashboard with real-time charts and prediction history
- Prometheus metrics exposed at `/metrics` for scraping
- Docker + Docker Compose for fully containerised deployment
- GitHub Actions CI — linting, model smoke test, server health check, Docker build verification

## Tech Stack

| Layer | Tool |
|---|---|
| Model | scikit-learn RandomForestClassifier |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + custom HTML/JS dashboard |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Project Structure
ml-monitored-endpoint/

├── .github/workflows/ci.yml   # CI pipeline

├── app/

│   ├── main.py                # FastAPI app, routes, metrics

│   ├── model.py               # Training and inference logic

│   ├── schemas.py             # Pydantic request/response models

│   └── templates/

│       └── dashboard.html     # Live monitoring dashboard

├── model/                     # Saved model artifact (git-ignored)

├── prometheus/

│   └── prometheus.yml         # Prometheus scrape config

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

## Quickstart

### Run with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Live dashboard | http://localhost:8000/dashboard |
| Health check | http://localhost:8000/health |
| Prometheus metrics | http://localhost:9090 |

### Run locally

```bash
pip install -r requirements.txt
python -m app.model
uvicorn app.main:app --reload
```

## API Reference

### `POST /predict`

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```json
{
  "prediction": 0,
  "label": "setosa",
  "confidence": 0.97,
  "probabilities": {
    "setosa": 0.97,
    "versicolor": 0.02,
    "virginica": 0.01
  },
  "latency_ms": 12.4
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "total_requests": 42
}
```

### `GET /api/stats`

Returns live monitoring data including latency history, prediction counts, error rate, and recent predictions.

### `GET /dashboard`

Live monitoring dashboard — updates every 3 seconds.

## Monitoring

The dashboard at `/dashboard` shows:

- Total requests, average latency, error rate, uptime
- Rolling latency line chart (last 30 predictions)
- Prediction class distribution doughnut chart
- Recent predictions table with confidence scores
- Interactive prediction form

Raw Prometheus metrics are available at `/metrics` and can be scraped by any Prometheus-compatible tool.
