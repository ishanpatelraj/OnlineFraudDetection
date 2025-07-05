import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import re

# Load model
with open("model.pkl", "rb") as f:
    models = pickle.load(f)

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load model columns
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

with open("amount_bins.pkl", "rb") as f:
    amount_bins = pickle.load(f)

st.title("💳 Fraud Detection Prediction")
st.markdown("Fill in the transaction details to get a fraud prediction.")

# Fields
categorical_fields = [
    'ProductCD',
    'card_network',
    'issuer_bank_code',
    'card_type',
    'card_id',
    'card_bin',
    'addr1',
    'addr2',
    'dist1',
    'dist2',
    'P_emaildomain',
    'R_emaildomain',
    'Operating_system',
    'Browser_type',
    'DeviceType',
    'DeviceInfo'
]


numerical_fields = [
    'TransactionID', 'TransactionAmt',
    # 'issuer_bank_code', 'card_bin',
    # 'addr1', 'addr2', 'dist1', 'dist2'
]

additional_numericals = [
    'recent_txn_count', 'card_usage_frequency', 'shared_device_count',
    'billing_address_usage', 'shipping_address_usage', 'device_browser_combo_count',
    'transaction_type_count', 'device_usage_frequency', 'inactive_device_count',
    'merchant_category_count', 'location_terminal_count',
    'rolling_txn_count_short_term', 'rolling_txn_count_mid_term', 'rolling_txn_count_long_term'
]

#string_fields = ['card_id', 'DeviceInfo']
input_fields = categorical_fields + numerical_fields

dropdown_options = {
    "ProductCD": ['H', 'C', 'S', 'R'],
    "card_network": ['mastercard', 'visa', 'american express', 'discover', 'nan'],
    "issuer_bank_code": [514.0, 100.0, 352.0, 375.0, 555.0],
    "card_type": ['credit', 'debit', 'charge card', 'nan'],
    "card_id": [4497, 2803, 16496, 4461, 1790, np.nan],
    "card_bin": [102, 226, 134, 224, 219, np.nan],
    "addr1": [420, 337, 170, 204, np.nan],
    "addr2": [87, 96, 35, 60, np.nan],
    "dist1": [np.nan],
    "dist2": [30.0,  98.0, 149.0,  84.0, np.nan],
    "P_emaildomain": ['gmail.com', 'anonymous.com', 'hotmail.com', 'aol.com', 'yahoo.com', 'nan'],
    "R_emaildomain": ['gmail.com', 'hotmail.com', 'outlook.com', 'anonymous.com', 'nan'],
    "Operating_system": ['Android 7.0', 'iOS 11.1.2', 'Mac OS X 10_11_6', 'Windows 10', 'nan'],
    "Browser_type": ['samsung browser 6.2', 'mobile safari 11.0', 'chrome 62.0', 'chrome 62.0 for android', 'nan'],
    "DeviceType": ['mobile', 'desktop', 'nan'],
    "DeviceInfo": ['SAMSUNG SM-G892A Build/NRD90M', 'iOS Device', 'Windows', 'MacOS', 'nan']
}

# Collect user input
data = {}
st.header("Transaction Input")
for field in categorical_fields:
    options = dropdown_options.get(field, ['nan'])
    selected = st.selectbox(f"{field}", options)
    data[field] = selected if selected != 'nan' else np.nan

for field in numerical_fields:
    data[field] = st.number_input(field, value=0.0)
# for field in string_fields:
#     data[field] = st.text_input(field, "NA")

input_date = st.date_input("Transaction Date", value=datetime.date(2024, 9, 8))
input_time = st.time_input("Transaction Time", value=datetime.time(0, 0))

# Combine date and time
input_datetime = datetime.datetime.combine(input_date, input_time)

# Reference date (should be same as used during training)
start_datetime = datetime.datetime.strptime('2024-09-08', "%Y-%m-%d")

# Calculate TransactionDT = seconds since reference date
transaction_dt = int((input_datetime - start_datetime).total_seconds())

data['TransactionDT'] = transaction_dt


if st.button("🔍 Predict Fraud"):
    df = pd.DataFrame([data])

    df['_P_emaildomain__addr1'] = df['P_emaildomain'].astype(str) + '__' + df['addr1'].astype(str)
    df['_card_id__issuer'] = df['card_id'].astype(str) + '__' + df['issuer_bank_code'].astype(str)
    df['_card_id__addr1'] = df['card_id'].astype(str) + '__' + df['addr1'].astype(str)
    df['_issuer__addr1'] = df['issuer_bank_code'].astype(str) + '__' + df['addr1'].astype(str)
    df['_cardid_issuer__addr1'] = df['_card_id__issuer'] + '__' + df['addr1'].astype(str)

    for col in ['_P_emaildomain__addr1', '_card_id__issuer', '_card_id__addr1', '_issuer__addr1', '_cardid_issuer__addr1']:
        if col + '_freq' in model_columns:
            df[col + '_freq'] = 0

    startDate = datetime.datetime.strptime('2024-09-08', "%Y-%m-%d")
    df['Date'] = df['TransactionDT'].apply(lambda x: startDate + datetime.timedelta(seconds=x))
    df['ymd'] = df['Date'].dt.year.astype(str) + '-' + df['Date'].dt.month.astype(str) + '-' + df['Date'].dt.day.astype(str)
    df['year_month'] = df['Date'].dt.year.astype(str) + '-' + df['Date'].dt.month.astype(str)
    df['weekday'] = df['Date'].dt.dayofweek
    df['hour'] = df['Date'].dt.hour
    df['day'] = df['Date'].dt.day
    df['_seq_day'] = df['TransactionDT'] // (24 * 60 * 60)
    df['_seq_week'] = df['_seq_day'] // 7
    df['_weekday_hour'] = df['weekday'].astype(str) + '_' + df['hour'].astype(str)
    df['_amount_qcut10'] = pd.cut(df['TransactionAmt'], bins=amount_bins, include_lowest=True)

    def get_time_bucket(hour):
        hour = int(hour)
        if 5 <= hour < 12: return 'morning'
        elif 12 <= hour < 17: return 'afternoon'
        elif 17 <= hour < 21: return 'evening'
        else: return 'night'

    df['_hour_bucket'] = df['hour'].apply(get_time_bucket)
    df['_is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    hour_distribution = {i: max(0.035, min(0.075, 0.035 + i * 0.001)) for i in range(24)}
    df['_hour_density'] = df['hour'].map(hour_distribution).fillna(0.04)

    hour_buckets = ['evening', 'night', 'morning']
    for bucket in hour_buckets:
        df[f'_hour_bucket_{bucket}'] = (df['_hour_bucket'] == bucket).astype(int)

    df.drop(columns=['_hour_bucket'], inplace=True)

    if '_amount_qcut10' in encoders:
        df['_amount_qcut10'] = df['_amount_qcut10'].astype(str).apply(
            lambda x: encoders['_amount_qcut10'].transform([x])[0]
            if x in encoders['_amount_qcut10'].classes_
            else -1
        )

    df.drop(columns=['Date'], inplace=True)

    d_features = [
        'days_since_prev_txn', 'days_since_first_txn', 'device_session_txn_gap',
        'txn_gap_same_card', 'txn_gap_same_billing_addr', 'days_since_last_login',
        'days_since_last_device_use', 'txn_gap_same_state', 'address_reuse_duration',
        'billing_shipping_time_diff', 'days_since_card_registration',
        'rolling_txn_time_short_term', 'rolling_txn_time_mid_term',
        'rolling_txn_time_long_term', 'rolling_txn_time_extended'
    ]

    interaction_fields = [
        '_weekday_hour', '_P_emaildomain__addr1', '_card_id__issuer',
        '_card_id__addr1', '_issuer__addr1', '_cardid_issuer__addr1'
    ]

    for col in interaction_fields:
        if col in df.columns and col in encoders:
            df[col] = df[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        elif col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    for col in d_features:
        df[col] = 0

    df['_amount_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)

    df['_amount_decimal_len'] = df['TransactionAmt'].apply(
        lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]) if '.' in str(x) else 0
    )

    df['_amount_fraction'] = df['TransactionAmt'].apply(
        lambda x: float('0.' + re.sub(r'^[0-9]|\\.|0+$', '', str(x))) if '.' in str(x) and re.search(r'\\d', str(x).split('.')[-1]) else 0.0
    )

    for col in categorical_fields + string_fields:
        if col in df.columns and col in encoders:
            df[col] = df[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    for col in additional_numericals:
        df[col] = 0

    df = df[model_columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    preds = np.mean([model.predict(df) for model in models], axis=0)
    pred_label = int(preds[0] > 0.5)

    st.subheader("Prediction Result")
    st.success(f"Fraud Prediction: {'FRAUD' if pred_label == 1 else 'LEGIT'} (prob = {preds[0]:.4f})")
