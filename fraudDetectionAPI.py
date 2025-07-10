from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import re
from pydantic import BaseModel

app = FastAPI()

with open("model.pkl", "rb") as f:
    models = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

with open("amount_bins.pkl", "rb") as f:
    amount_bins = pickle.load(f)

categorical_fields = [
    'ProductCD', 'card_network', 'issuer_bank_code', 'card_type',
    'card_id', 'card_bin', 'addr1', 'addr2', 'dist1', 'dist2',
    'P_emaildomain', 'R_emaildomain', 'Operating_system',
    'Browser_type', 'DeviceType', 'DeviceInfo'
]

numerical_fields = ['TransactionID', 'TransactionAmt']

additional_numericals = [
    'recent_txn_count', 'card_usage_frequency', 'shared_device_count',
    'billing_address_usage', 'shipping_address_usage', 'device_browser_combo_count',
    'transaction_type_count', 'device_usage_frequency', 'inactive_device_count',
    'merchant_category_count', 'location_terminal_count',
    'rolling_txn_count_short_term', 'rolling_txn_count_mid_term', 'rolling_txn_count_long_term'
]

class TransactionInput(BaseModel):
    ProductCD: str
    card_network: str
    issuer_bank_code: float
    card_type: str
    card_id: int
    card_bin: float
    addr1: float
    addr2: float
    dist1: float
    dist2: float
    P_emaildomain: str
    R_emaildomain: str
    Operating_system: str
    Browser_type: str
    DeviceType: str
    DeviceInfo: str
    TransactionID: float
    TransactionAmt: float
    TransactionDT: int

@app.post("/predict")
async def predict_transaction(payload: TransactionInput):
    data = payload.dict()
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
    df['weekday'] = df['Date'].dt.dayofweek
    df['hour'] = df['Date'].dt.hour
    df['_weekday_hour'] = df['weekday'].astype(str) + '_' + df['hour'].astype(str)
    df['_amount_qcut10'] = pd.cut(df['TransactionAmt'], bins=amount_bins, include_lowest=True)

    def get_time_bucket(hour):
        if 5 <= hour < 12: return 'morning'
        elif 12 <= hour < 17: return 'afternoon'
        elif 17 <= hour < 21: return 'evening'
        else: return 'night'

    df['_hour_bucket'] = df['hour'].apply(get_time_bucket)
    df['_is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    hour_distribution = {i: max(0.035, min(0.075, 0.035 + i * 0.001)) for i in range(24)}
    df['_hour_density'] = df['hour'].map(hour_distribution).fillna(0.04)

    for bucket in ['evening', 'night', 'morning']:
        df[f'_hour_bucket_{bucket}'] = (df['_hour_bucket'] == bucket).astype(int)
    df.drop(columns=['_hour_bucket', 'Date'], inplace=True)

    if '_amount_qcut10' in encoders:
        df['_amount_qcut10'] = df['_amount_qcut10'].astype(str).apply(
            lambda x: encoders['_amount_qcut10'].transform([x])[0] if x in encoders['_amount_qcut10'].classes_ else -1
        )

    for col in [
        'days_since_prev_txn', 'days_since_first_txn', 'device_session_txn_gap',
        'txn_gap_same_card', 'txn_gap_same_billing_addr', 'days_since_last_login',
        'days_since_last_device_use', 'txn_gap_same_state', 'address_reuse_duration',
        'billing_shipping_time_diff', 'days_since_card_registration',
        'rolling_txn_time_short_term', 'rolling_txn_time_mid_term',
        'rolling_txn_time_long_term', 'rolling_txn_time_extended'
    ]:
        df[col] = 0

    for col in ['_weekday_hour', '_P_emaildomain__addr1', '_card_id__issuer', '_card_id__addr1', '_issuer__addr1', '_cardid_issuer__addr1']:
        if col in df.columns and col in encoders:
            df[col] = df[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        elif col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    df['_amount_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
    df['_amount_decimal_len'] = df['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]) if '.' in str(x) else 0)
    df['_amount_fraction'] = df['TransactionAmt'].apply(lambda x: float('0.' + re.sub(r'^[0-9]|\.|0+$', '', str(x))) if '.' in str(x) and re.search(r'\d', str(x).split('.')[-1]) else 0.0)

    for col in categorical_fields:
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
    return {"fraud_prediction": int(preds[0] > 0.5), "fraud_probability": float(preds[0])}

@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API! Use POST /predict with JSON input."}