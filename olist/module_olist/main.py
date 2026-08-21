from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features

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


if __name__ == "__main__":
    main()