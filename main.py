import sys
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).parent
RAW_PATH    = BASE_DIR / "data" / "raw" / "Online_Retail.xlsx"
DATA_PATH   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
EXPORT_PATH = BASE_DIR / "exports"
SCHEMA_PATH = BASE_DIR / "exports" / "galaxy_schema"

sys.path.insert(0, str(BASE_DIR / "scripts"))

from data_cleaning       import clean_retail
from rmf_analysis        import customer_segmentation
from geographic_analysis import country_revenue, segment_by_country
from product_analysis    import product_affinity
from cohort_analysis     import cohort_matrix, cohort_long
from galaxy_schema_export import (dim_date, dim_customer, dim_segment,
                                  dim_product, fact_sales, fact_returns)
from quality_checks      import run_checks


def run():
    EXPORT_PATH.mkdir(exist_ok=True)
    SCHEMA_PATH.mkdir(exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Cleaning raw data...")
    df_clean = clean_retail(pd.read_excel(RAW_PATH))
    df_clean.to_csv(DATA_PATH, index=False)

    # Purchase-behaviour analyses run on sales only; returns are modelled separately.
    df_sales   = df_clean[df_clean['TransactionType'] == 'Sale']
    df_returns = df_clean[df_clean['TransactionType'] == 'Return']
    print(f"  {len(df_sales):,} sales, {len(df_returns):,} returns")

    print("Running RFM segmentation...")
    df_rfm = customer_segmentation(df_sales).sort_values('RFM_Score', ascending=False)
    df_rfm.to_csv(EXPORT_PATH / 'customer_rmf.csv', index=False)

    print("Running geographic analysis...")
    country_revenue(df_sales).to_csv(EXPORT_PATH / 'country_revenue.csv', index=False)
    segment_by_country(df_sales, df_rfm).to_csv(EXPORT_PATH / 'segment_by_country.csv', index=False)

    print("Running product analysis...")
    product_affinity(df_sales, df_rfm, top_n=10).to_csv(EXPORT_PATH / 'product_affinity.csv', index=False)

    print("Running cohort analysis...")
    df_cohort, df_retention = cohort_matrix(df_sales)
    df_cohort.to_csv(EXPORT_PATH / 'cohort_counts.csv', index=False)
    df_retention.to_csv(EXPORT_PATH / 'cohort_retention.csv', index=False)
    cohort_long(df_cohort, df_retention).to_csv(EXPORT_PATH / 'cohort_retention_long.csv', index=False)

    print("Generating galaxy schema exports...")
    dim_date(df_clean).to_csv(SCHEMA_PATH / 'dim_date.csv', index=False)
    dim_customer(df_clean, df_rfm).to_csv(SCHEMA_PATH / 'dim_customer.csv', index=False)
    dim_segment().to_csv(SCHEMA_PATH / 'dim_segment.csv', index=False)
    dim_product(df_clean).to_csv(SCHEMA_PATH / 'dim_product.csv', index=False)
    fact_sales(df_sales).to_csv(SCHEMA_PATH / 'fact_sales.csv', index=False)
    fact_returns(df_returns).to_csv(SCHEMA_PATH / 'fact_returns.csv', index=False)

    print("\nRunning quality checks...")
    ok = run_checks()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    run()
