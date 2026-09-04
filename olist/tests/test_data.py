import pandas as pd

from module_olist.modeling.cross_validation import cross_validate_models
from module_olist.modeling.split import FEATURES, split_data


def _build_demo_data(n_per_class: int = 60) -> pd.DataFrame:
    rows = []
    for label in [0, 1]:
        for i in range(n_per_class):
            rows.append(
                {
                    "purchase_hour": (i + label) % 24,
                    "purchase_weekday": (i + label) % 7,
                    "purchase_month": (i + label) % 12 + 1,
                    "promised_days": 3 + ((i + label) % 10),
                    "item_count": 1 + ((i + label) % 4),
                    "seller_count": 1 + ((i + label) % 3),
                    "total_price": round(50 + (i * 1.5) + label * 25, 2),
                    "total_freight": round(5 + (i * 0.4) + label * 3, 2),
                    "customer_state": "SP" if label == 0 else "RJ",
                    "is_late": label,
                }
            )

    return pd.DataFrame(rows)


def test_split_data_returns_expected_shapes():
    data = _build_demo_data(n_per_class=60)

    X_train, X_test, y_train, y_test = split_data(data)

    assert list(X_train.columns) == FEATURES
    assert X_train.shape[0] == 96
    assert X_test.shape[0] == 24
    assert y_train.shape[0] == 96
    assert y_test.shape[0] == 24
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_cross_validate_models_returns_selected_model_and_threshold():
    data = _build_demo_data(n_per_class=80)
    X_train, _, y_train, _ = split_data(data)

    model_name, threshold = cross_validate_models(X_train, y_train)

    assert model_name in {
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
    }
    assert 0.01 <= threshold <= 0.99
