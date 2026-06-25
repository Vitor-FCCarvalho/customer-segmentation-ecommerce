import pandas as pd
import os
from pathlib import Path

# Service / non-merchandise stock codes
SERVICE_CODES = {'POST', 'DOT', 'C2', 'M', 'm', 'S', 'B', 'BANK CHARGES',
                 'AMAZONFEE', 'CRUK', 'D'}

# Short codes for compact region slicers (ISO-2 where it applies; UK/EIRE/RSA etc.
# follow the dataset's own naming rather than strict ISO).
COUNTRY_CODE = {
    'Australia': 'AU', 'Austria': 'AT', 'Bahrain': 'BH', 'Belgium': 'BE',
    'Brazil': 'BR', 'Canada': 'CA', 'Channel Islands': 'CHI', 'Cyprus': 'CY',
    'Czech Republic': 'CZ', 'Denmark': 'DK', 'EIRE': 'IE', 'European Community': 'EU',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hong Kong': 'HK', 'Iceland': 'IS', 'Israel': 'IL', 'Italy': 'IT',
    'Japan': 'JP', 'Lebanon': 'LB', 'Lithuania': 'LT', 'Malta': 'MT',
    'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
    'RSA': 'ZA', 'Saudi Arabia': 'SA', 'Singapore': 'SG', 'Spain': 'ES',
    'Sweden': 'SE', 'Switzerland': 'CH', 'USA': 'US', 'United Arab Emirates': 'AE',
    'United Kingdom': 'UK', 'Unspecified': 'UNS', 'Unknown': 'UNK',
}

SEGMENT_CATALOG = [
    # (Segment, SortOrder, Description, RecommendedAction)
    ('Champions',           1, 'Bought recently, buy often and spend the most.',
     'Reward themby giving them access to new products, loyalty perks, and referral programs.'),
    ('Loyal Customers',     2, 'Buy regularly and recently, just below top spenders.',
     'Upsell higher value products, ask for reviews, invite to loyalty program.'),
    ('Potential Loyalists', 3, 'Recent customers with average frequency.',
     'Membership offers and personalised recommendations to increase frequency.'),
    ('New Customers',       4, 'Bought very recently for the first time.',
     'Onboarding emails, first-purchase follow-up, build the relationship early.'),
    ('Promising',           5, 'Recent customers who have not bought much yet.',
     'Brand awareness content and small incentives on a second purchase.'),
    ('Need Attention',      6, 'Above-average recency and frequency, but slipping.',
     'Limited-time offers and reactivation reminders based on past purchases.'),
    ('About to Sleep',      7, 'Below-average recency, low frequency.',
     'Share popular products and renewed-interest discounts before they lapse.'),
    ('At Risk',             8, 'Used to buy often, have not purchased in a long time.',
     'Win-back campaign: personalised reactivation emails with strong offers.'),
    ("Can't Lose Them",     9, 'Were among the most frequent buyers, now inactive.',
     'High-touch win-back: call or premium offer; do not lose them to competitors.'),
    ('Hibernating',        10, 'Long since last purchase, low past engagement.',
     'Low-cost reactivation: seasonal newsletters and relevant discounts.'),
    ('Lost',               11, 'Lowest recency and frequency scores.',
     'Minimal spend: include in broad campaigns only, attempt one revival offer.'),
    ('Unknown',            12, 'No customer identifier recorded.',
     'Not addressable for CRM campaigns, track volume.'),
]


def dim_date(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df['InvoiceDate'])

    start = pd.Timestamp(year=d.min().year, month=d.min().month, day=1)
    end   = pd.Timestamp(year=d.max().year, month=12, day=31)
    df_date = pd.DataFrame({'Date': pd.date_range(start, end, freq='D')})

    df_date['DateKey']   = df_date['Date'].dt.strftime('%Y%m%d').astype(int)
    df_date['Year']      = df_date['Date'].dt.year
    df_date['Quarter']   = df_date['Date'].dt.quarter
    df_date['Month']     = df_date['Date'].dt.month
    df_date['MonthName'] = df_date['Date'].dt.strftime('%B')
    df_date['MonthYear'] = df_date['Date'].dt.strftime('%b-%Y') 
    df_date['MonthYearKey'] = df_date['Year'] * 100 + df_date['Month']
    df_date['DayOfWeek'] = df_date['Date'].dt.dayofweek + 1 
    df_date['DayName']   = df_date['Date'].dt.strftime('%A')

    return df_date[['DateKey', 'Date', 'Year', 'Quarter', 'Month', 'MonthName',
                    'MonthYear', 'MonthYearKey','DayOfWeek', 'DayName']]

def dim_customer(df_clean: pd.DataFrame, df_rfm: pd.DataFrame) -> pd.DataFrame:
    df_known = df_clean.dropna(subset=['CustomerID']).copy()
    df_known['CustomerID']  = df_known['CustomerID'].astype(int)
    df_known['InvoiceDate'] = pd.to_datetime(df_known['InvoiceDate'])

    # Country from all activity; first/last purchase from sales only
    df_country = (df_known.groupby('CustomerID')['Country']
                          .agg(lambda x: x.value_counts().index[0])
                          .reset_index())

    df_sales = df_known[df_known['TransactionType'] == 'Sale']
    df_life  = (df_sales.groupby('CustomerID')
                        .agg(FirstPurchaseDate=('InvoiceDate', 'min'),
                             LastPurchaseDate=('InvoiceDate', 'max'))
                        .reset_index())
    df_life['FirstPurchaseDate'] = df_life['FirstPurchaseDate'].dt.normalize()
    df_life['LastPurchaseDate']  = df_life['LastPurchaseDate'].dt.normalize()
    df_life['CohortMonth']       = df_life['FirstPurchaseDate'].dt.strftime('%Y-%m')

    df_attrs = df_country.merge(df_life, on='CustomerID', how='left')

    df_rfm = df_rfm.copy()
    df_rfm['CustomerID'] = df_rfm['CustomerID'].astype(int)
    df_rfm['AvgOrderValue'] = (df_rfm['Monetary'] / df_rfm['Frequency']).round(2)

    # Left join RFM so return-only customers survive
    df_customers = df_attrs.merge(df_rfm, on='CustomerID', how='left')
    df_customers['Segment'] = df_customers['Segment'].fillna('Unknown')

    # Compact region code for slicers; fall back to first 3 letters if unmapped
    df_customers['CountryCode'] = (df_customers['Country'].map(COUNTRY_CODE)
                                   .fillna(df_customers['Country'].str[:3].str.upper()))

    # Sentinel row for transactions with no CustomerID
    unknown = pd.DataFrame([{
        'CustomerID': -1, 'Recency': None, 'Monetary': None, 'Frequency': None,
        'R_Score': None, 'F_Score': None, 'M_Score': None, 'RFM_Score': None,
        'Segment': 'Unknown', 'AvgOrderValue': None, 'Country': 'Unknown',
        'CountryCode': COUNTRY_CODE['Unknown'],
        'FirstPurchaseDate': None, 'LastPurchaseDate': None, 'CohortMonth': None
    }])

    cols = ['CustomerID', 'Recency', 'Monetary', 'Frequency', 'R_Score', 'F_Score',
            'M_Score', 'RFM_Score', 'Segment', 'AvgOrderValue', 'Country', 'CountryCode',
            'FirstPurchaseDate', 'LastPurchaseDate', 'CohortMonth']
    return pd.concat([unknown, df_customers], ignore_index=True)[cols]

def dim_segment() -> pd.DataFrame:
    return pd.DataFrame(SEGMENT_CATALOG,
                        columns=['Segment', 'SortOrder', 'Description', 'RecommendedAction'])

def dim_product(df: pd.DataFrame) -> pd.DataFrame:
    # Use most frequent description per StockCode to resolve minor naming inconsistencies
    df_prod = (df.dropna(subset=['Description'])
                 .groupby('StockCode', as_index=False)['Description']
                 .agg(lambda x: x.value_counts().index[0])
    )
    is_voucher = df_prod['StockCode'].astype(str).str.lower().str.startswith('gift')
    df_prod['ProductType'] = 'Merchandise'
    df_prod.loc[df_prod['StockCode'].isin(SERVICE_CODES), 'ProductType'] = 'Postage & Fees'
    df_prod.loc[is_voucher, 'ProductType'] = 'Gift Voucher'

    return df_prod

def _fact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['CustomerID']  = df['CustomerID'].fillna(-1).astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['DateKey']     = df['InvoiceDate'].dt.strftime('%Y%m%d').astype(int)
    df['Hour']        = df['InvoiceDate'].dt.hour
    df['Quantity']    = df['Quantity'].abs()
    df['Revenue']     = (df['Quantity'] * df['UnitPrice']).round(2)

    return df[['InvoiceNo', 'StockCode', 'CustomerID', 'DateKey', 'Hour', 'Quantity', 'UnitPrice', 'Revenue']]

def fact_sales(df_sales: pd.DataFrame) -> pd.DataFrame:
    return _fact(df_sales)

def fact_returns(df_returns: pd.DataFrame) -> pd.DataFrame:
    return _fact(df_returns)

if __name__ == "__main__":
    BASE_DIR  = Path(__file__).parent.parent
    data_path = BASE_DIR / "data" / "clean" / "clean_retail.csv"
    rfm_path  = BASE_DIR / "exports" / "customer_rmf.csv"
    out_path  = BASE_DIR / "exports" / "galaxy_schema"

    out_path.mkdir(exist_ok=True)

    df_clean = pd.read_csv(data_path, low_memory=False)
    df_rfm   = pd.read_csv(rfm_path)

    df_sales   = df_clean[df_clean['TransactionType'] == 'Sale']
    df_returns = df_clean[df_clean['TransactionType'] == 'Return']

    dim_date(df_clean).to_csv(out_path / 'dim_date.csv', index=False)
    dim_customer(df_clean, df_rfm).to_csv(out_path / 'dim_customer.csv', index=False)
    dim_segment().to_csv(out_path / 'dim_segment.csv', index=False)
    dim_product(df_clean).to_csv(out_path / 'dim_product.csv', index=False)
    fact_sales(df_sales).to_csv(out_path / 'fact_sales.csv', index=False)
    fact_returns(df_returns).to_csv(out_path / 'fact_returns.csv', index=False)
