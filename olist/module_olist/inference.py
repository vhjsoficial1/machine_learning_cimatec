import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from module_olist.modeling.predict import predict_from_artifact


DEFAULT_INPUT_PATH = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "orders_predictions.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"


def run_inference(
	input_path: Path = DEFAULT_INPUT_PATH,
	output_path: Path = DEFAULT_OUTPUT_PATH,
	model_path: Path = DEFAULT_MODEL_PATH,
	metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
	"""Executa a inferência em um dataset refinado e salva suas predições."""
	data = pd.read_csv(input_path)
	predictions = predict_from_artifact(
		data=data,
		model_path=model_path,
		metadata_path=metadata_path,
	)

	result_columns = ["order_id"] if "order_id" in data.columns else []
	result = pd.concat(
		[data[result_columns].reset_index(drop=True), predictions.reset_index(drop=True)],
		axis=1,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(output_path, index=False)
	logger.success(f"Predições salvas em: {output_path}")
	return result


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Executa inferência no modelo treinado.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_inference(
		input_path=args.input,
		output_path=args.output,
		model_path=args.model,
		metadata_path=args.metadata,
	)
