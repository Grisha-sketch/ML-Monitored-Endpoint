from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1)
    sepal_width: float = Field(..., gt=0, example=3.5)
    petal_length: float = Field(..., gt=0, example=1.4)
    petal_width: float = Field(..., gt=0, example=0.2)


class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_requests: int
    