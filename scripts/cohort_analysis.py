import pandas as pd
import os
from pathlib import Path


def cohort_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=['CustomerID']).copy()
    df['CustomerID']  = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['OrderMonth']  = df['InvoiceDate'].dt.to_period('M')

    df_first = df.groupby('CustomerID')['OrderMonth'].min().reset_index()
    df_first.columns = ['CustomerID', 'CohortMonth']

    df = df.merge(df_first, on='CustomerID')
    df['MonthsElapsed'] = (df['OrderMonth'] - df['CohortMonth']).apply(lambda x: x.n)

    df_cohort = (df.groupby(['CohortMonth', 'MonthsElapsed'])['CustomerID']
                   .nunique()
                   .unstack(fill_value=0)
    )

    cohort_sizes = df_cohort.iloc[:, 0]
    df_retention = (df_cohort.div(cohort_sizes, axis=0) * 100).round(1)

    df_cohort.index    = df_cohort.index.astype(str)
    df_retention.index = df_retention.index.astype(str)

    df_cohort.columns    = [f'Month_{c}'     for c in df_cohort.columns]
    df_retention.columns = [f'Month_{c}_Pct' for c in df_retention.columns]

    return df_cohort.reset_index(), df_retention.reset_index()

if __name__ == "__main__":
    BASE_DIR = Path.cwd().parent
    data_path   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    export_path = BASE_DIR / "exports"

    df_clean = pd.read_csv(data_path, low_memory=False)

    df_cohort, df_retention = cohort_matrix(df_clean)

    df_cohort.to_csv(os.path.join(export_path, 'cohort_counts.csv'), index=False)
    df_retention.to_csv(os.path.join(export_path, 'cohort_retention.csv'), index=False)
