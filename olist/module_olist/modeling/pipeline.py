from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ↑ utilizar esses modelos para uso de features e targets ↑

# gradientboosting faz a árvore da sequência de forma que cada árvore corrige os erros da anterior, é um modelo de boosting

# diferença entre gradientboosting, xgboost e lightgbm é que o xgboost é mais rápido e eficiente, enquanto o lightgbm é mais rápido e eficiente em grandes datasets, já o gradientboosting é mais simples e fácil de interpretar, mas pode ser mais lento e menos eficiente em grandes datasets

# xgboost é uma implementação otimizada do algoritmo de gradient boosting, que utiliza técnicas de regularização e paralelização para melhorar a performance e reduzir o overfitting. Ele é amplamente utilizado em competições de machine learning devido à sua alta precisão e eficiência.

# lightgbm é uma biblioteca de aprendizado de máquina baseada em gradient boosting, que utiliza uma abordagem de histogramas para acelerar o treinamento e reduzir o uso de memória. Ele é especialmente eficiente em grandes conjuntos de dados e é capaz de lidar com variáveis categóricas diretamente, sem a necessidade de codificação one-hot.

NUMERICAL_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight"
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state"
]

# montar esteira de pre-processamento
def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # aplicar transformações para features numéricas
            ("numeric", "passthrough", NUMERICAL_FEATURES),
            # aplicar transformações para features categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo Gradient Boosting.
    """
    preprocessor = create_preprocessor()
    model = GradientBoostingClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42 # semente para reproduzibilidade
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )

def create_xgboost_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo XGBoost.
    """
    preprocessor = create_preprocessor()
    model = XGBClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42, # semente para reproduzibilidade
        use_label_encoder=False, # desabilitar o uso do codificador de rótulos
        eval_metric="logloss" # métrica de avaliação
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )

def create_lightgbm_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo LightGBM.
    """
    preprocessor = create_preprocessor()
    model = LGBMClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42 # semente para reproduzibilidade
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )
