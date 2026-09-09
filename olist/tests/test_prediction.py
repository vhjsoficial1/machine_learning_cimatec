import json

import joblib
import pandas as pd

from module_olist.inference import run_inference
from module_olist.modeling.predict import predict
from module_olist.modeling.split import FEATURES


class _DemoModel:
    def predict_proba(self, data):
        probabilities = data["total_price"].to_numpy() / 100
        return pd.DataFrame({0: 1 - probabilities, 1: probabilities}).to_numpy()


def _build_features():
    return pd.DataFrame(
        [
            [10, 2, 1, 7, 1, 1, 20, 3, "SP"],
            [11, 3, 2, 8, 1, 1, 80, 4, "RJ"],
        ],
        columns=FEATURES,
    )


def test_predict_applies_threshold_to_positive_probability():
    predictions = predict(_DemoModel(), _build_features(), threshold=0.5)

    assert list(predictions.columns) == [
        "late_probability",
        "is_late_prediction",
    ]
    assert predictions["is_late_prediction"].tolist() == [0, 1]


def test_run_inference_loads_artifacts_and_saves_predictions(tmp_path):
    input_path = tmp_path / "orders.csv"
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    output_path = tmp_path / "predictions.csv"

    data = _build_features().assign(order_id=["order-1", "order-2"], is_late=[0, 1])
    data.to_csv(input_path, index=False)
    joblib.dump(_DemoModel(), model_path)
    metadata_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    result = run_inference(input_path, output_path, model_path, metadata_path)

    assert output_path.exists()
    assert list(result.columns) == [
        "order_id",
        "late_probability",
        "is_late_prediction",
    ]
    assert "is_late" not in result.columns