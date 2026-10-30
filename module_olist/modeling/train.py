import pandas as pd
from modeling.pipeline import (create_gradient_boosting_pipeline, create_lightgbm_pipeline, 
                      create_preprocessor, create_xgboost_pipeline)
import joblib
import json
from loguru import logger

def create_selected_model(model_name):
    """
    Cria o pipeline correspondente
    ao modelo selecionado.
    """

    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline,
        "XGBoost": create_xgboost_pipeline,
        "LightGBM": create_lightgbm_pipeline,
    }

    if model_name not in pipelines:
        raise ValueError(
            f"Modelo desconhecido: {model_name}"
        )

    return pipelines[model_name]()

def train_model(model_name, threshold, X_train: pd.DataFrame, y_train: pd.Series, model_path, metadata_path):
    """
    Trains the best model
    found during cross validation
    and saves it.
    """
    model = create_selected_model(model_name)
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.success(f"Model saved in {model_path}")

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "selection_metric": "pr_auc",
        "threshold_metric": "f1",
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(metadata, file, indent=4)

    logger.success(f"Metadata saved in {metadata_path}")
    
    return model