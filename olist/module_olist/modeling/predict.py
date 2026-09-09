import json
from pathlib import Path

import joblib
import pandas as pd

from module_olist.modeling.split import FEATURES


def load_model(model_path: Path):
	"""Carrega o pipeline de modelo salvo pelo treinamento."""
	return joblib.load(model_path)


def load_threshold(metadata_path: Path) -> float:
	"""Carrega e valida o threshold escolhido durante a validação."""
	with metadata_path.open(encoding="utf-8") as file:
		metadata = json.load(file)

	threshold = float(metadata["threshold"])
	if not 0 < threshold < 1:
		raise ValueError("O threshold precisa estar entre 0 e 1.")

	return threshold


def predict(
	model,
	data: pd.DataFrame,
	threshold: float,
) -> pd.DataFrame:
	"""Gera probabilidade de atraso e classe prevista para cada pedido."""
	missing_features = sorted(set(FEATURES) - set(data.columns))
	if missing_features:
		raise ValueError(
			f"Dados de predição sem as features obrigatórias: {missing_features}"
		)

	if not 0 < threshold < 1:
		raise ValueError("O threshold precisa estar entre 0 e 1.")

	probabilities = model.predict_proba(data[FEATURES])[:, 1]

	return pd.DataFrame(
		{
			"late_probability": probabilities,
			"is_late_prediction": (probabilities >= threshold).astype("int8"),
		},
		index=data.index,
	)


def predict_from_artifact(
	data: pd.DataFrame,
	model_path: Path,
	metadata_path: Path,
) -> pd.DataFrame:
	"""Carrega o modelo treinado e gera predições usando seu threshold."""
	model = load_model(model_path)
	threshold = load_threshold(metadata_path)
	return predict(model, data, threshold)
