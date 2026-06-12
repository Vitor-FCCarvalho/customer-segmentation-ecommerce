import pandas as pd
import os
from pathlib import Path


def country_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df_country = (df.groupby('Country', as_index=False)
                    .agg(
                        Revenue=('Revenue', 'sum'),
                        Orders=('InvoiceNo', 'nunique'),
                        Customers=('CustomerID', 'nunique')
                    )
                    .sort_values('Revenue', ascending=False)
    )
    df_country['RevenueShare'] = (df_country['Revenue'] / df_country['Revenue'].sum() * 100).round(2)

    return df_country

def segment_by_country(df_clean: pd.DataFrame, df_rfm: pd.DataFrame) -> pd.DataFrame:
    df = df_clean[['CustomerID', 'Country']].dropna(subset=['CustomerID']).drop_duplicates()
    df['CustomerID'] = df['CustomerID'].astype(int)

    df_rfm = df_rfm.copy()
    df_rfm['CustomerID'] = df_rfm['CustomerID'].astype(int)

    df_merged = df.merge(df_rfm[['CustomerID', 'Segment']], on='CustomerID', how='inner')

    df_pivot = (df_merged.groupby(['Country', 'Segment'])
                         .size()
                         .unstack(fill_value=0)
                         .reset_index()
    )
    df_pivot['Total'] = df_pivot.iloc[:, 1:].sum(axis=1)
    df_pivot = df_pivot.sort_values('Total', ascending=False)

    return df_pivot

if __name__ == "__main__":
    BASE_DIR = Path.cwd().parent
    data_path   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    rfm_path    = BASE_DIR / "exports" / "customer_rmf.csv"
    export_path = BASE_DIR / "exports"

    df_clean = pd.read_csv(data_path, low_memory=False)
    df_rfm   = pd.read_csv(rfm_path)

    df_country  = country_revenue(df_clean)
    df_segments = segment_by_country(df_clean, df_rfm)

    df_country.to_csv(os.path.join(export_path, 'country_revenue.csv'), index=False)
    df_segments.to_csv(os.path.join(export_path, 'segment_by_country.csv'), index=False)
