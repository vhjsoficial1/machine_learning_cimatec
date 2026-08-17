import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("../data/raw/olist_orders_dataset.csv")

profile = ProfileReport(df, title="Relatório de profiling - orders", explorative=True)
profile.to_file("../reports/profiling_orders.html")