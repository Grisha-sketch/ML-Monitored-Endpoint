import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "model/model.pkl"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def train_and_save():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained — accuracy: {accuracy:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No model found, training now...")
        train_and_save()
    return joblib.load(MODEL_PATH)


def predict(model, features: list[float]) -> dict:
    X = np.array(features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0].tolist()
    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "label": CLASS_NAMES[prediction],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(p, 4)
            for i, p in enumerate(probabilities)
        }
    }


if __name__ == "__main__":
    train_and_save()
    