from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.split import split_data
from module_olist.modeling.train import train_model

def main() -> None:
    orders_path = RAW_DATA_DIR / "olist_orders_dataset.csv"
    items_path = RAW_DATA_DIR / "olist_order_items_dataset.csv"
    customers_path = RAW_DATA_DIR / "olist_customers_dataset.csv"

    orders, items, customers = load_data(
        orders_path=orders_path,
        items_path=items_path,
        customers_path=customers_path,
    )

    data = create_dataset(orders=orders, items=items, customers=customers)
    data = create_features(data)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / "olist_interim_dataset.csv"
    save_dataset(dataset=data, output_path=output_path)

    X_train, X_test, y_train, y_test = split_data(data)
    models = train_model(X_train=X_train, y_train=y_train)
    evaluate_models(models=models, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()