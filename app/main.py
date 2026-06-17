import time
import threading
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import load_model, predict
from app.schemas import PredictRequest, PredictResponse, HealthResponse

model = None
total_requests = 0
error_count = 0
start_time = time.time()
latency_window = deque(maxlen=30)
prediction_counts = {"setosa": 0, "versicolor": 0, "virginica": 0}
recent_predictions = deque(maxlen=6)
lock = threading.Lock()

templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    print("Model loaded successfully")
    yield


app = FastAPI(
    title="ML Monitored Endpoint",
    description="Iris classifier with live monitoring dashboard",
    version="1.0.0",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        total_requests=total_requests
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    global total_requests, error_count

    features = [
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]

    t0 = time.time()
    try:
        result = predict(model, features)
    except Exception as e:
        with lock:
            error_count += 1
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = round((time.time() - t0) * 1000, 2)

    with lock:
        total_requests += 1
        latency_window.append(latency_ms)
        prediction_counts[result["label"]] += 1
        recent_predictions.appendleft({
            "id": total_requests,
            "sepal_length": request.sepal_length,
            "sepal_width": request.sepal_width,
            "petal_length": request.petal_length,
            "petal_width": request.petal_width,
            "label": result["label"],
            "confidence": result["confidence"],
            "latency_ms": latency_ms
        })

    return PredictResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=latency_ms
    )


@app.get("/api/stats")
def stats():
    with lock:
        uptime_seconds = int(time.time() - start_time)
        avg_latency = (
            round(sum(latency_window) / len(latency_window), 2)
            if latency_window else 0.0
        )
        err_rate = (
            round((error_count / (total_requests + error_count)) * 100, 2)
            if (total_requests + error_count) > 0 else 0.0
        )
        return {
            "total_requests": total_requests,
            "avg_latency_ms": avg_latency,
            "error_rate_pct": err_rate,
            "uptime_seconds": uptime_seconds,
            "latency_history": list(latency_window),
            "prediction_counts": dict(prediction_counts),
            "recent_predictions": list(recent_predictions)
        }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})