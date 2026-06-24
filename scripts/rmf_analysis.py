import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import date


def recency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df_recency = df.groupby('CustomerID', as_index=False)['InvoiceDate'].max()
    df_recency.columns = ['CustomerID', 'MostRecentPurchaseDate']
    df_recency['MostRecentPurchaseDate'] = df_recency['MostRecentPurchaseDate'].astype('datetime64[us]')
    last_date  = df_recency['MostRecentPurchaseDate'].max()
    df_recency['Recency'] = df_recency['MostRecentPurchaseDate'].apply(lambda x: (last_date - x).days)

    return df_recency

def monetary_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df_monetary   = df.groupby('CustomerID', as_index=False)['Revenue'].sum()
    df_monetary.columns = ['CustomerID', 'Monetary']

    return df_monetary

def frequency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df_frequency = df.groupby('CustomerID', as_index=False)['InvoiceNo'].nunique()
    df_frequency.columns = ['CustomerID', 'Frequency']

    return df_frequency

def rmf(df: pd.DataFrame) -> pd.DataFrame:
    df_recency   = recency_analysis(df)
    df_monetary  = monetary_analysis(df)
    df_frequency = frequency_analysis(df)

    df_rmf = (df_recency.merge(df_monetary, on='CustomerID')
                        .merge(df_frequency, on='CustomerID')
                        .drop(columns=['MostRecentPurchaseDate'])
    )
    df_rmf['CustomerID'] = df_rmf['CustomerID'].astype(int)

    return df_rmf

def customer_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = rmf(df)

    df['R_Score'] = pd.qcut(df['Recency'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_Score'] = pd.qcut(np.log1p(df['Monetary']).rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    return df

def assign_rfm_segment(r: int, f: int, m: int) -> str:
    
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 4:
        return 'Loyal Customers'
    elif r >= 4 and f >= 2:
        return 'Potential Loyalists'
    elif r == 5:
        return 'New Customers'
    elif r == 4:
        return 'Promising'
    elif r == 3 and f >= 2:
        return 'Need Attention'
    elif r == 3:
        return 'About to Sleep'
    elif f == 5:
        return "Can't Lose Them"
    elif f >= 3:
        return 'At Risk'
    elif r == 2:
        return 'Hibernating'
    else:
        return 'Lost'

def customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    df = customer_scoring(df)
    df['Segment'] = df.apply(
        lambda row: assign_rfm_segment(row['R_Score'], row['F_Score'], row['M_Score']), axis=1
    )

    return df

if __name__ == "__main__":
    BASE_DIR = Path.cwd().parent
    data_path   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    export_path = BASE_DIR / "exports"

    df = pd.read_csv(data_path, low_memory=False)
    df = df[df['TransactionType'] == 'Sale']  # RFM is purchase behaviour; exclude returns

    df_rmf = customer_segmentation(df)
    df_rmf = df_rmf.sort_values('RFM_Score', ascending=False)

    df_rmf.to_csv(os.path.join(export_path, r'customer_rmf.csv'), index=False)
