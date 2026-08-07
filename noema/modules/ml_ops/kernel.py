"""ML Ops module — training pipelines, model serving, experiment tracking."""

from dataclasses import dataclass, field
from typing import Any

MODEL_SUGGESTIONS = {
    "classification": {
        "sklearn": [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
            "SVC",
        ],
        "pytorch": ["MLPClassifier", "CNN1D"],
        "tensorflow": ["DNNClassifier", "CNN"],
        "huggingface": ["bert-base-uncased (fine-tuned)", "distilbert-base-uncased"],
        "algorithms": ["Random Forest", "XGBoost", "LightGBM", "Neural Network"],
    },
    "regression": {
        "sklearn": ["RandomForestRegressor", "GradientBoostingRegressor", "Ridge", "Lasso"],
        "pytorch": ["MLPRegressor"],
        "tensorflow": ["DNNRegressor"],
        "huggingface": ["bert-base-uncased (regression head)"],
        "algorithms": ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"],
    },
    "nlp": {
        "sklearn": ["TF-IDF + SVM", "TF-IDF + Naive Bayes"],
        "pytorch": ["LSTM", "Transformer"],
        "tensorflow": ["BiLSTM", "Transformer"],
        "huggingface": [
            "bert-base-uncased",
            "roberta-base",
            "gpt2",
            "t5-small",
            "distilbert-base-uncased",
        ],
        "algorithms": ["BERT", "RoBERTa", "GPT", "T5", "Word2Vec + Classical ML"],
    },
    "cv": {
        "pytorch": ["ResNet", "EfficientNet", "VisionTransformer", "YOLOv8"],
        "tensorflow": ["ResNet50", "MobileNetV2", "EfficientNetB0"],
        "huggingface": ["vit-base-patch16-224", "google/vit-base-patch16-224"],
        "algorithms": [
            "CNN (ResNet/EfficientNet)",
            "Vision Transformer (ViT)",
            "YOLO (Object Detection)",
        ],
    },
    "recommendation": {
        "sklearn": ["NearestNeighbors", "SVD"],
        "pytorch": ["NeuralCF", "WideAndDeep"],
        "tensorflow": ["WideAndDeep", "TwoTower"],
        "huggingface": [],
        "algorithms": [
            "Collaborative Filtering",
            "Content-Based",
            "Hybrid",
            "Matrix Factorization",
        ],
    },
}


@dataclass
class ExperimentConfig:
    name: str = ""
    model: str = ""
    dataset: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)


class MLPipeline:
    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def define_pipeline(self, steps: list[dict[str, Any]]) -> "MLPipeline":
        self._steps = steps
        return self

    def suggest_model(self, task_type: str, framework: str = "sklearn") -> dict[str, Any]:
        task_lower = task_type.lower()
        suggestions = MODEL_SUGGESTIONS.get(task_lower, {})
        framework_models = suggestions.get(framework, suggestions.get("sklearn", []))
        all_algorithms = suggestions.get("algorithms", [])

        return {
            "task_type": task_lower,
            "framework": framework,
            "suggested_models": framework_models,
            "algorithms": all_algorithms,
            "recommendation": framework_models[0]
            if framework_models
            else "No specific recommendation",
        }

    def to_code(
        self, framework: str = "sklearn", experiment: ExperimentConfig | None = None
    ) -> str:
        exp = experiment or ExperimentConfig(name="experiment", model="auto", dataset="data.csv")
        fw = framework.lower()

        if fw == "sklearn":
            return self._to_sklearn(exp)
        elif fw == "pytorch":
            return self._to_pytorch(exp)
        elif fw == "tensorflow":
            return self._to_tensorflow(exp)
        elif fw == "huggingface":
            return self._to_huggingface(exp)
        return self._to_sklearn(exp)

    def _to_sklearn(self, exp: ExperimentConfig) -> str:
        return f'''"""ML Experiment: {exp.name} — scikit-learn"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import mlflow.sklearn


def load_data(path: str = "{exp.dataset}"):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators={exp.hyperparameters.get("n_estimators", 100)},
            max_depth={exp.hyperparameters.get("max_depth", "None")},
            random_state=42,
            n_jobs=-1,
        )),
    ])

    with mlflow.start_run(run_name="{exp.name}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Accuracy: {{acc:.4f}}")
        print(report)

        joblib.dump(pipeline, "{exp.name}_model.pkl")
        return pipeline


if __name__ == "__main__":
    X, y = load_data()
    model = train(X, y)
'''

    def _to_pytorch(self, exp: ExperimentConfig) -> str:
        return f'''"""ML Experiment: {exp.name} — PyTorch"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow
import mlflow.pytorch


class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train(epochs: int = {exp.hyperparameters.get("epochs", 50)}, lr: float = {exp.hyperparameters.get("lr", 0.001)}):
    X_train = torch.randn(1000, 64)
    y_train = torch.randint(0, 10, (1000,))

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size={exp.hyperparameters.get("batch_size", 32)}, shuffle=True)

    model = Net(input_dim=64, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name="{exp.name}"):
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            mlflow.log_metric("loss", total_loss / len(loader), step=epoch)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {{epoch+1}}/{{epochs}}, Loss: {{total_loss/len(loader):.4f}}")

        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), "{exp.name}_model.pth")
        return model


if __name__ == "__main__":
    train()
'''

    def _to_tensorflow(self, exp: ExperimentConfig) -> str:
        return f'''"""ML Experiment: {exp.name} — TensorFlow/Keras"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow


def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate={exp.hyperparameters.get("lr", 0.001)}),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = {exp.hyperparameters.get("epochs", 50)}):
    X_train = np.random.randn(1000, 64).astype(np.float32)
    y_train = np.random.randint(0, 10, (1000,))

    model = build_model(input_dim=64, num_classes=10)

    with mlflow.start_run(run_name="{exp.name}"):
        mlflow.tensorflow.autolog()

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size={exp.hyperparameters.get("batch_size", 32)},
            validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(patience=5)],
        )

        model.save("{exp.name}_model.keras")
        return model


if __name__ == "__main__":
    train()
'''

    def _to_huggingface(self, exp: ExperimentConfig) -> str:
        return f'''"""ML Experiment: {exp.name} — Hugging Face Transformers"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate


def train():
    model_name = "{exp.model or "distilbert-base-uncased"}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("{exp.dataset or "imdb"}")
    metric = evaluate.load("accuracy")

    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized = dataset.map(preprocess, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="{exp.name}_results",
        num_train_epochs={exp.hyperparameters.get("epochs", 3)},
        per_device_train_batch_size={exp.hyperparameters.get("batch_size", 8)},
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="{exp.name}_logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("{exp.name}_model")
    tokenizer.save_pretrained("{exp.name}_model")
    return model


if __name__ == "__main__":
    train()
'''


class ModelServing:
    def generate_serving_code(
        self, model_type: str = "classification", framework: str = "sklearn"
    ) -> str:
        return f'''"""Model Serving — {model_type} model ({framework})"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model Server", version="1.0.0")

model = None


class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "default"


class PredictionResponse(BaseModel):
    prediction: any
    confidence: Optional[float] = None
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            confidence = float(np.max(proba))

        return PredictionResponse(
            prediction=prediction[0].tolist() if hasattr(prediction[0], "tolist") else prediction[0],
            confidence=confidence,
            model_name=request.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for req in requests:
        features = np.array(req.features).reshape(1, -1)
        prediction = model.predict(features)
        results.append({{
            "prediction": prediction[0].tolist() if hasattr(prediction[0], "tolist") else prediction[0],
            "model_name": req.model_name,
        }})
    return results
'''

    def generate_endpoint(self, api_type: str = "rest") -> str:
        if api_type.lower() == "grpc":
            return """syntax = "proto3";

service ModelService {
  rpc Predict(PredictionRequest) returns (PredictionResponse);
  rpc BatchPredict(stream PredictionRequest) returns (stream PredictionResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
}

message PredictionRequest {
  repeated float features = 1;
  string model_name = 2;
}

message PredictionResponse {
  int32 prediction = 1;
  float confidence = 2;
  string model_name = 3;
}

message HealthRequest {}
message HealthResponse {
  string status = 1;
  bool model_loaded = 2;
}
"""
        else:
            return self.generate_serving_code()


class MLOpsModule:
    NAME = "ml_ops"
    DESCRIPTION = (
        "ML Ops: training pipelines, model serving, experiment tracking, model suggestions"
    )

    def __init__(self) -> None:
        self.pipeline = MLPipeline()
        self.serving = ModelServing()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "suggest"
        if "train" in str(task_title).lower() or "train" in task_tags:
            action = "train"
        elif "serve" in str(task_title).lower() or "serving" in task_tags or "deploy" in task_tags:
            action = "serve"
        elif "experiment" in str(task_title).lower() or "experiment" in task_tags:
            action = "experiment"
        elif "endpoint" in str(task_title).lower() or "api" in task_tags:
            action = "endpoint"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        task_type = metadata.get("task_type", "classification")
        framework = metadata.get("framework", "sklearn")

        if action == "suggest":
            suggestion = self.pipeline.suggest_model(task_type, framework)
            return {
                "action": "suggest_model",
                **suggestion,
                "_confidence": 0.80,
            }
        elif action == "train":
            experiment = ExperimentConfig(
                name=metadata.get("experiment_name", "experiment"),
                model=metadata.get("model", "auto"),
                dataset=metadata.get("dataset", "data.csv"),
                hyperparameters=metadata.get("hyperparameters", {}),
                metrics=metadata.get("metrics", ["accuracy"]),
            )
            code = self.pipeline.to_code(framework, experiment)
            return {
                "action": "training_pipeline",
                "framework": framework,
                "experiment": {
                    "name": experiment.name,
                    "model": experiment.model,
                    "dataset": experiment.dataset,
                },
                "code": code,
                "_confidence": 0.85,
            }
        elif action == "serve":
            model_type = str(metadata.get("model_type", task_type))
            serving_code = self.serving.generate_serving_code(model_type, framework)
            return {
                "action": "model_serving",
                "model_type": model_type,
                "code": serving_code,
                "_confidence": 0.85,
            }
        elif action == "endpoint":
            api_type = metadata.get("api_type", "rest")
            endpoint = self.serving.generate_endpoint(api_type)
            return {
                "action": "endpoint",
                "api_type": api_type,
                "code": endpoint,
                "_confidence": 0.85,
            }

        suggestion = self.pipeline.suggest_model(task_type, framework)
        return {
            "action": "suggest",
            **suggestion,
            "_confidence": 0.70,
        }
