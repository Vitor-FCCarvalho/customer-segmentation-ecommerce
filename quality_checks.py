import sys
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data" / "clean" / "clean_retail.csv"
EXPORT_PATH = BASE_DIR / "exports"
SCHEMA_PATH = BASE_DIR / "exports" / "star_schema"

VALID_SEGMENTS = {
    'Champions', 'Loyal Customers', 'Potential Loyalists',
    'New Customers', 'Promising', 'Need Attention',
    'About to Sleep', 'At Risk', "Can't Lose Them",
    'Hibernating', 'Lost'
}

EXPECTED_SOURCE_COLS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country',
                        'TransactionType']

EXPECTED_RFM_COLS    = ['CustomerID', 'Recency', 'Monetary', 'Frequency',
                        'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']

EXPECTED_SCHEMA_FILES = ['fact_sales.csv', 'fact_returns.csv', 'dim_customer.csv',
                         'dim_date.csv', 'dim_product.csv', 'dim_segment.csv']


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)

def _report(label: str, passed: bool, detail: str = '') -> bool:
    status = 'PASS' if passed else 'FAIL'
    line   = f'  [{status}]  {label}'
    if detail:
        line += f' — {detail}'
    print(line)
    return passed

def check_source_exists() -> bool:
    return _report('clean_retail.csv exists', DATA_PATH.exists())

def check_source_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in EXPECTED_SOURCE_COLS if c not in df.columns]
    return _report('Expected source columns present',
                   not missing, f'missing: {missing}' if missing else '')

def check_source_row_count(df: pd.DataFrame) -> bool:
    n = len(df)
    ok = 525_000 <= n <= 545_000
    return _report('Source row count in expected range [525k–545k]', ok, f'{n:,} rows')

def check_transaction_quantity_signs(df: pd.DataFrame) -> bool:
    bad_sales   = ((df['TransactionType'] == 'Sale')   & (df['Quantity'] <= 0)).sum()
    bad_returns = ((df['TransactionType'] == 'Return') & (df['Quantity'] >= 0)).sum()
    ok = bad_sales == 0 and bad_returns == 0
    detail = '' if ok else f'{bad_sales} sales <=0, {bad_returns} returns >=0'
    return _report('Sales have positive and returns negative Quantity', ok, detail)

def check_returns_are_c_invoices(df: pd.DataFrame) -> bool:
    returns = df[df['TransactionType'] == 'Return']
    non_c = (~returns['InvoiceNo'].astype(str).str.startswith('C')).sum()
    return _report("All returns are 'C'-prefixed invoices",
                   non_c == 0, f'{non_c} non-C returns' if non_c else '')

def check_no_negative_price(df: pd.DataFrame) -> bool:
    n = (df['UnitPrice'] <= 0).sum()
    return _report('No non-positive UnitPrice values', n == 0, f'{n:,} found' if n else '')

def check_rfm_exists() -> bool:
    return _report('customer_rmf.csv exists', (EXPORT_PATH / 'customer_rmf.csv').exists())

def check_rfm_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in EXPECTED_RFM_COLS if c not in df.columns]
    return _report('Expected RFM columns present',
                   not missing, f'missing: {missing}' if missing else '')

def check_rfm_score_range(df: pd.DataFrame) -> bool:
    for col in ['R_Score', 'F_Score', 'M_Score']:
        out = df[col].dropna()
        if out.min() < 1 or out.max() > 5:
            return _report(f'{col} values in range [1–5]', False,
                           f'min={out.min()}, max={out.max()}')
    return _report('R/F/M scores in range [1–5]', True)

def check_rfm_score_calculation(df: pd.DataFrame) -> bool:
    calculated = df['R_Score'] + df['F_Score'] + df['M_Score']
    mismatch = (calculated != df['RFM_Score']).sum()
    return _report('RFM_Score == R_Score + F_Score + M_Score',
                   mismatch == 0, f'{mismatch} mismatches' if mismatch else '')

def check_rfm_no_nulls(df: pd.DataFrame) -> bool:
    nulls = df[EXPECTED_RFM_COLS].isnull().sum()
    bad   = nulls[nulls > 0]
    return _report('No null values in RFM columns',
                   bad.empty, bad.to_dict() if not bad.empty else '')

def check_rfm_valid_segments(df: pd.DataFrame) -> bool:
    unexpected = set(df['Segment'].dropna().unique()) - VALID_SEGMENTS
    return _report('All segment labels are valid',
                   not unexpected, f'unexpected: {unexpected}' if unexpected else '')

def check_rfm_segment_distribution(df: pd.DataFrame) -> bool:
    max_share = df['Segment'].value_counts(normalize=True).max()
    ok = max_share <= 0.60
    return _report('No single segment exceeds 60% of customers',
                   ok, f'largest share: {max_share:.1%}')


def check_cohort_exists() -> bool:
    return _report('cohort_retention.csv exists',
                   (EXPORT_PATH / 'cohort_retention.csv').exists())

def check_cohort_month0(df: pd.DataFrame) -> bool:
    col = 'Month_0_Pct'
    if col not in df.columns:
        return _report(f'{col} column present in cohort retention', False)
    not_100 = (df[col] != 100.0).sum()
    return _report('Month_0_Pct is 100% for every cohort',
                   not_100 == 0, f'{not_100} cohorts with unexpected value' if not_100 else '')

def check_cohort_retention_range(df: pd.DataFrame) -> bool:
    pct_cols = [c for c in df.columns if c.endswith('_Pct')]
    vals = df[pct_cols].values
    out_of_range = ((vals < 0) | (vals > 100)).sum()
    return _report('All retention values in range [0–100]',
                   out_of_range == 0, f'{out_of_range} out-of-range values' if out_of_range else '')

def check_cohort_long_exists() -> bool:
    return _report('cohort_retention_long.csv exists',
                   (EXPORT_PATH / 'cohort_retention_long.csv').exists())

def check_cohort_long_consistency(df: pd.DataFrame) -> bool:
    month0 = df[df['MonthsElapsed'] == 0]
    bad = ((month0['ActiveCustomers'] != month0['CohortSize']) |
           (month0['RetentionPct'] != 100.0)).sum()
    return _report('Long cohort table: month 0 equals cohort size at 100%',
                   bad == 0, f'{bad} inconsistent rows' if bad else '')


def check_schema_files_exist() -> bool:
    missing = [f for f in EXPECTED_SCHEMA_FILES if not (SCHEMA_PATH / f).exists()]
    return _report('All star schema files exist',
                   not missing, f'missing: {missing}' if missing else '')

def check_fk_customer(fact: pd.DataFrame, dim: pd.DataFrame, name: str) -> bool:
    valid_ids = set(dim['CustomerID'].unique())
    orphans   = ~fact['CustomerID'].isin(valid_ids)
    return _report(f'{name} -> dim_customer FK integrity',
                   not orphans.any(), f'{orphans.sum():,} orphaned rows' if orphans.any() else '')

def check_fk_date(fact: pd.DataFrame, dim: pd.DataFrame, name: str) -> bool:
    valid_keys = set(dim['DateKey'].unique())
    orphans    = ~fact['DateKey'].isin(valid_keys)
    return _report(f'{name} -> dim_date FK integrity',
                   not orphans.any(), f'{orphans.sum():,} orphaned rows' if orphans.any() else '')

def check_fk_product(fact: pd.DataFrame, dim: pd.DataFrame, name: str) -> bool:
    valid_codes = set(dim['StockCode'].unique())
    orphans     = ~fact['StockCode'].isin(valid_codes)
    return _report(f'{name} -> dim_product FK integrity',
                   not orphans.any(), f'{orphans.sum():,} orphaned rows' if orphans.any() else '')

def check_fk_segment(dim_cust: pd.DataFrame, dim_seg: pd.DataFrame) -> bool:
    valid    = set(dim_seg['Segment'].unique())
    orphans  = ~dim_cust['Segment'].isin(valid)
    return _report('dim_customer -> dim_segment FK integrity',
                   not orphans.any(), f'{orphans.sum():,} orphaned rows' if orphans.any() else '')

def check_unknown_sentinel(dim: pd.DataFrame) -> bool:
    return _report('Unknown sentinel row (CustomerID = -1) present in dim_customer',
                   -1 in dim['CustomerID'].values)

def check_fact_row_count(fact: pd.DataFrame, expected: int, name: str) -> bool:
    ok = len(fact) == expected
    return _report(f'{name} row count matches clean data',
                   ok, f'fact={len(fact):,}, expected={expected:,}')

def check_returns_positive_magnitude(fact: pd.DataFrame) -> bool:
    bad = ((fact['Quantity'] < 0) | (fact['Revenue'] < 0)).sum()
    return _report('fact_returns stored as positive magnitudes',
                   bad == 0, f'{bad} negative rows' if bad else '')

def check_dim_date_contiguous(dim: pd.DataFrame) -> bool:
    # Verify the number of rows equals the number of calendar days between first and last date.
    dates = pd.to_datetime(dim['Date']).sort_values()
    expected = (dates.max() - dates.min()).days + 1
    ok = len(dates) == expected and dates.is_unique
    detail = '' if ok else f'{expected - len(dates)} missing days'
    return _report('dim_date is a contiguous daily calendar', ok, detail)

def check_dim_product_unique_key(dim: pd.DataFrame) -> bool:
    
    dup_cs = dim['StockCode'].duplicated().sum()
    dup_ci = dim['StockCode'].astype(str).str.upper().duplicated().sum()
    ok = dup_cs == 0 and dup_ci == 0
    detail = '' if ok else f'{dup_cs} exact, {dup_ci} case-insensitive duplicates'
    return _report('dim_product StockCode is a unique key', ok, detail)

def check_sales_invoices_numeric(fact: pd.DataFrame) -> bool:

    bad = (~fact['InvoiceNo'].astype(str).str.fullmatch(r'\d+')).sum()
    return _report('fact_sales InvoiceNo values are all numeric',
                   bad == 0, f'{bad} non-numeric invoices' if bad else '')

def run_checks() -> bool:
    results = []

    print('\n[Source Data]')
    if not check_source_exists():
        print('  Cannot continue! Clean_retail.csv is missing.')
        return False
    df_source = _load(DATA_PATH)
    results += [
        check_source_columns(df_source),
        check_source_row_count(df_source),
        check_transaction_quantity_signs(df_source),
        check_returns_are_c_invoices(df_source),
        check_no_negative_price(df_source),
    ]
    n_sales   = (df_source['TransactionType'] == 'Sale').sum()
    n_returns = (df_source['TransactionType'] == 'Return').sum()

    print('\n[RFM Export]')
    if not check_rfm_exists():
        results.append(False)
    else:
        df_rfm = _load(EXPORT_PATH / 'customer_rmf.csv')
        results += [
            check_rfm_columns(df_rfm),
            check_rfm_score_range(df_rfm),
            check_rfm_score_calculation(df_rfm),
            check_rfm_no_nulls(df_rfm),
            check_rfm_valid_segments(df_rfm),
            check_rfm_segment_distribution(df_rfm),
        ]

    print('\n[Cohort Export]')
    if not check_cohort_exists():
        results.append(False)
    else:
        df_cohort = _load(EXPORT_PATH / 'cohort_retention.csv')
        results += [
            check_cohort_month0(df_cohort),
            check_cohort_retention_range(df_cohort),
        ]
    if not check_cohort_long_exists():
        results.append(False)
    else:
        df_cohort_long = _load(EXPORT_PATH / 'cohort_retention_long.csv')
        results.append(check_cohort_long_consistency(df_cohort_long))

    print('\n[Star Schema Integrity]')
    if not check_schema_files_exist():
        results.append(False)
    else:
        fact_sales   = _load(SCHEMA_PATH / 'fact_sales.csv')
        fact_returns = _load(SCHEMA_PATH / 'fact_returns.csv')
        dim_cust = _load(SCHEMA_PATH / 'dim_customer.csv')
        dim_dt   = _load(SCHEMA_PATH / 'dim_date.csv')
        dim_prod = _load(SCHEMA_PATH / 'dim_product.csv')
        dim_seg  = _load(SCHEMA_PATH / 'dim_segment.csv')
        results += [
            check_fk_customer(fact_sales, dim_cust, 'fact_sales'),
            check_fk_date(fact_sales, dim_dt, 'fact_sales'),
            check_fk_product(fact_sales, dim_prod, 'fact_sales'),
            check_fk_customer(fact_returns, dim_cust, 'fact_returns'),
            check_fk_date(fact_returns, dim_dt, 'fact_returns'),
            check_fk_product(fact_returns, dim_prod, 'fact_returns'),
            check_returns_positive_magnitude(fact_returns),
            check_sales_invoices_numeric(fact_sales),
            check_dim_product_unique_key(dim_prod),
            check_dim_date_contiguous(dim_dt),
            check_fk_segment(dim_cust, dim_seg),
            check_unknown_sentinel(dim_cust),
            check_fact_row_count(fact_sales, n_sales, 'fact_sales'),
            check_fact_row_count(fact_returns, n_returns, 'fact_returns'),
        ]

    passed = sum(results)
    total  = len(results)
    print(f'\n{passed}/{total} checks passed.\n')
    return passed == total


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
