import numpy as np
from tensorflow.keras.models import load_model

from src.common.utils import load_config, load_pickle, logger
from src.common.database import get_collection
from src.common.schema import TIME_COLUMN, META_COLUMN
from src.preprocess import run_preprocess

def load_models(config):
    models = {}
    run_models = config["inference"]["run_models"]

    if "rf" in run_models:
        rf_path = config["model"]["rf"]["path"]
        models["rf"] = load_pickle(rf_path)
        logger.info("Loaded Random Forest model")

    if "lstm" in run_models:
        lstm_path = config["model"]["lstm"]["path"]
        models["lstm"] = load_model(lstm_path)
        logger.info("Loaded LSTM model")

    return models

def reshape_for_lstm(X):
    return np.expand_dims(X, axis=1)

def run_inference():
    logger.info("Starting inference...")
    config = load_config()
    pred_col = get_collection(config, "prediction_collection")

    X_scaled, df_processed = run_preprocess()

    if X_scaled is None or len(X_scaled) == 0:
        logger.warning("No data for inference")
        return

    models = load_models(config)
    results = []

    for model_name, model in models.items():
        logger.info(f"Running model: {model_name}")

        if model_name == "rf":
            preds = model.predict(X_scaled)
        elif model_name == "lstm":
            X_lstm = reshape_for_lstm(X_scaled)
            preds = model.predict(X_lstm).flatten()
        else:
            continue

        for i, pred in enumerate(preds):
            record = {
                TIME_COLUMN: df_processed.iloc[i][TIME_COLUMN].to_pydatetime(),
                META_COLUMN: df_processed.iloc[i][META_COLUMN],
                "model": model_name,
                "prediction": float(pred)
            }
            results.append(record)

    if results:
        pred_col.insert_many(results)
        logger.info(f"Inserted {len(results)} predictions")

    logger.info("Inference done.")

if __name__ == "__main__":
    run_inference()