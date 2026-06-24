import pandas as pd
import os
from pathlib import Path

from star_schema_export import SERVICE_CODES


def product_affinity(df_clean: pd.DataFrame, df_rfm: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = df_clean.dropna(subset=['CustomerID', 'Description']).copy()
    df = df[~df['StockCode'].isin(SERVICE_CODES)]  # merchandise only
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['Revenue']    = df['Quantity'] * df['UnitPrice']

    df_rfm = df_rfm.copy()
    df_rfm['CustomerID'] = df_rfm['CustomerID'].astype(int)

    df_merged = df.merge(df_rfm[['CustomerID', 'Segment']], on='CustomerID', how='inner')

    df_agg = (df_merged.groupby(['Segment', 'StockCode', 'Description'], as_index=False)
                       .agg(Revenue=('Revenue', 'sum'),
                            Quantity=('Quantity', 'sum'),
                            Orders=('InvoiceNo', 'nunique'),
                            Customers=('CustomerID', 'nunique'))
                       .sort_values(['Segment', 'Revenue'], ascending=[True, False])
    )
    df_agg['Revenue'] = df_agg['Revenue'].round(2)

    df_top = (df_agg.groupby('Segment').head(top_n).reset_index(drop=True))

    return df_top

if __name__ == "__main__":
    BASE_DIR = Path.cwd().parent
    data_path   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    rfm_path    = BASE_DIR / "exports" / "customer_rmf.csv"
    export_path = BASE_DIR / "exports"

    df_clean = pd.read_csv(data_path, low_memory=False)
    df_clean = df_clean[df_clean['TransactionType'] == 'Sale']  # product affinity = purchases only
    df_rfm   = pd.read_csv(rfm_path)

    df_top_products = product_affinity(df_clean, df_rfm, top_n=10)
    df_top_products.to_csv(os.path.join(export_path, 'product_affinity.csv'), index=False)
