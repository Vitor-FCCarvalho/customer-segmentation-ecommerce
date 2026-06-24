"""Raw -> clean transformation for the Online Retail dataset.

Cleaning rules
--------------
* Drop exact duplicate rows.
* Drop unit-price errors (`UnitPrice <= 0`): zero-price give-aways.
* Classify by invoice number:
    - purely numeric invoice  -> Sale   (must have positive quantity)
    - 'C' + numeric invoice   -> Return (always negative quantity)
* Drop everything else. This removes:
    - non-'C' negative-quantity rows  -> inventory write-offs (no CustomerID),
    - 'A' invoices ("Adjust bad debt") -> accounting adjustments, not customer
      transactions (one carries a positive price and would otherwise leak into
      sales, leaving the only non-numeric InvoiceNo in fact_sales).

Keeping sale invoices purely numeric also lets Power BI infer InvoiceNo's type
cleanly instead of choking on a stray alphabetic prefix.
"""

import pandas as pd
from pathlib import Path

RAW_COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
               'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']


def clean_retail(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.drop_duplicates().copy()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str).str.strip().str.upper()

    is_sale_invoice   = df['InvoiceNo'].str.fullmatch(r'\d+')          # e.g. 536365
    is_return_invoice = df['InvoiceNo'].str.fullmatch(r'C\d+')         # e.g. C536379

    valid_price = df['UnitPrice'] > 0
    valid_row   = valid_price & (
        (is_return_invoice & (df['Quantity'] < 0)) |   # returns: 'C' invoice, qty < 0
        (is_sale_invoice   & (df['Quantity'] > 0))     # sales:   numeric invoice, qty > 0
    )

    df = df[valid_row].copy()
    df['TransactionType'] = df['InvoiceNo'].str.startswith('C').map({True: 'Return', False: 'Sale'})

    return df.sort_values('InvoiceDate').reset_index(drop=True)


if __name__ == "__main__":
    BASE_DIR  = Path(__file__).parent.parent
    raw_path  = BASE_DIR / "data" / "raw" / "Online_Retail.xlsx"
    out_path  = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_path} ...")
    df_raw = pd.read_excel(raw_path)

    df_clean = clean_retail(df_raw)

    n_sales   = (df_clean['TransactionType'] == 'Sale').sum()
    n_returns = (df_clean['TransactionType'] == 'Return').sum()
    print(f"Cleaned {len(df_raw):,} raw rows -> {len(df_clean):,} clean rows "
          f"({n_sales:,} sales, {n_returns:,} returns).")

    df_clean.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
