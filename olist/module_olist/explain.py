from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import shap

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import (
    load_model,
)


def main():

    logger.info("Iniciando explicabilidade do modelo...")

    # Carrega o dataset
    data = pd.read_csv(INTERIM_DATA_DIR /
        "orders_dataset_refined.csv"
    )

    # Remove target das entradas
    X = data.drop(columns=["is_late"])

    # Seleciona algumas amostras
    X_sample = X.sample(n=min(100, len(X)), random_state=42)

    # Carrega o modelo treinado
    model = load_model(
        model_path=(
            MODELS_DIR /
            "best_model.joblib"
        ),
    )

    logger.info("Modelo carregado com sucesso.")

    # Aplica o pré-processamento
    X_transformed = (
        prepare_data_for_shap(
            pipeline=model,
            X=X_sample,
        )
    )

    # Cria o explicador
    explainer = create_explainer(
        pipeline=model
    )

    # Calcula SHAP
    shap_values = (
        calculate_shap_values(
            explainer=explainer,
            X_transformed=X_transformed,
        )
    )

    logger.success(
        "Valores SHAP calculados."
    )
    
    shap.plots.bar(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
         FIGURES_DIR /
        "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    sample_position = 0
    
    shap.plots.waterfall(
        shap_values[
            sample_position
        ],
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.success(
        "Gráficos SHAP gerados com sucesso."
    )

    logger.info(
        f"Gráficos salvos em: "
        f"{FIGURES_DIR}"
    )

if __name__ == "__main__":
    main()