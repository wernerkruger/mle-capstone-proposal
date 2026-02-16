"""
Data preprocessing for Arvato customer segmentation and campaign response prediction.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Common missing/unknown codes in Arvato/DIAS demographic data
MISSING_CODES = [-1, 0, -2, -9, 9]


def load_data(data_dir='data', nrows_azdias=None, nrows_customers=None):
    """Load all Arvato datasets. Use nrows_* to load subsets for faster iteration (clustering only)."""
    azdias = pd.read_csv(f'{data_dir}/Udacity_AZDIAS_052018.csv', sep=';', low_memory=False, nrows=nrows_azdias)
    customers = pd.read_csv(f'{data_dir}/Udacity_CUSTOMERS_052018.csv', sep=';', low_memory=False, nrows=nrows_customers)
    mailout_train = pd.read_csv(f'{data_dir}/Udacity_MAILOUT_052018_TRAIN.csv', sep=';', low_memory=False)
    mailout_test = pd.read_csv(f'{data_dir}/Udacity_MAILOUT_052018_TEST.csv', sep=';', low_memory=False)
    return azdias, customers, mailout_train, mailout_test


def get_customer_only_columns(customers, azdias):
    """Get columns in CUSTOMERS that are not in AZDIAS (metadata columns to drop)."""
    cust_cols = set(customers.columns)
    azdias_cols = set(azdias.columns)
    return list(cust_cols - azdias_cols)


def replace_missing_codes(df):
    """Replace known missing/unknown codes with NaN."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
    return df


def clean_dataframe(df, drop_cols=None, max_col_missing=0.4, max_row_missing=0.3):
    """Clean dataframe: replace missing codes, drop high-missing columns/rows."""
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    df = replace_missing_codes(df)

    col_missing = df.isnull().mean()
    cols_to_drop = col_missing[col_missing > max_col_missing].index.tolist()
    df = df.drop(columns=cols_to_drop, errors='ignore')

    row_missing = df.isnull().mean(axis=1)
    df = df[row_missing <= max_row_missing].copy()

    return df


def _encode_column(series, encoder=None):
    """Encode a single column. Returns encoded series and encoder."""
    mask = series.notna()
    vals = series.loc[mask].astype(str)
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(vals.unique())
    classes = set(encoder.classes_)
    transformed = np.array([encoder.transform([v])[0] if v in classes else -1 for v in vals])
    out = np.full(len(series), np.nan)
    out[mask] = transformed
    return pd.Series(out, index=series.index), encoder


def encode_and_impute(df, fit_imputer=None, fit_encoders=None):
    """Encode categorical variables and impute missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    encoders = fit_encoders or {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        le = encoders.get(col)
        encoded, le = _encode_column(df[col], le)
        df[col] = encoded
        encoders[col] = le

    all_cols = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
    df_work = df[all_cols].copy()
    df_work = df_work.astype(float)

    imputer = fit_imputer or SimpleImputer(strategy='median')
    arr = imputer.fit_transform(df_work) if fit_imputer is None else imputer.transform(df_work)
    df_clean = pd.DataFrame(arr, columns=all_cols, index=df.index)

    return df_clean, imputer, encoders


def preprocess_pipeline(azdias, customers, mailout_train, mailout_test,
                       sample_azdias=None, sample_customers=None):
    """Full preprocessing pipeline."""
    extra_cols = get_customer_only_columns(customers, azdias)

    azdias_clean = clean_dataframe(azdias, drop_cols=extra_cols)
    customers_clean = clean_dataframe(customers, drop_cols=extra_cols)

    if 'RESPONSE' in mailout_train.columns:
        y_train = mailout_train['RESPONSE']
        mailout_train = mailout_train.drop(columns=['RESPONSE'])
    else:
        y_train = None

    common_cols = sorted(set(azdias_clean.columns) & set(customers_clean.columns) &
                        set(mailout_train.columns) & set(mailout_test.columns))

    azdias_align = azdias_clean[common_cols].copy()
    customers_align = customers_clean[common_cols].copy()
    mailout_train_align = mailout_train[common_cols].copy()
    mailout_test_align = mailout_test[common_cols].copy()

    azdias_encoded, imputer, encoders = encode_and_impute(azdias_align)
    customers_encoded, _, _ = encode_and_impute(customers_align, fit_imputer=imputer, fit_encoders=encoders)
    mailout_train_encoded, _, _ = encode_and_impute(mailout_train_align, fit_imputer=imputer, fit_encoders=encoders)
    mailout_test_encoded, _, _ = encode_and_impute(mailout_test_align, fit_imputer=imputer, fit_encoders=encoders)

    if sample_azdias:
        azdias_encoded = azdias_encoded.sample(n=min(sample_azdias, len(azdias_encoded)), random_state=42)
    if sample_customers:
        customers_encoded = customers_encoded.sample(n=min(sample_customers, len(customers_encoded)), random_state=42)

    return {
        'azdias': azdias_encoded,
        'customers': customers_encoded,
        'mailout_train': mailout_train_encoded,
        'mailout_test': mailout_test_encoded,
        'y_train': y_train,
        'imputer': imputer,
        'encoders': encoders,
        'feature_columns': common_cols,
    }
