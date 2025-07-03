from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import json
import os

app = Flask(__name__)

# Load model
with open("lgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load model columns
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # You must create templates/index.html

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Collect input values from form
        input_fields = [
            'TransactionAmt', 'ProductCD', 'card_id', 'issuer_bank_code', 'card_network',
            'card_bin', 'card_type', 'addr1', 'addr2', 'dist1', 'dist2',
            'P_emaildomain', 'R_emaildomain', 'recent_txn_count', 'card_usage_frequency',
            'shared_device_count', 'billing_address_usage', 'shipping_address_usage',
            'device_browser_combo_count', 'transaction_type_count', 'device_usage_frequency',
            'inactive_device_count', 'merchant_category_count', 'location_terminal_count',
            'rolling_txn_count_short_term', 'rolling_txn_count_mid_term', 'rolling_txn_count_long_term',
            'days_since_prev_txn', 'days_since_first_txn', 'device_session_txn_gap',
            'txn_gap_same_card', 'txn_gap_same_billing_addr', 'days_since_last_login',
            'days_since_last_device_use', 'txn_gap_same_state', 'address_reuse_duration',
            'billing_shipping_time_diff', 'days_since_card_registration',
            'rolling_txn_time_short_term', 'rolling_txn_time_mid_term',
            'rolling_txn_time_long_term', 'rolling_txn_time_extended',
            '_seq_day', '_seq_week', '_weekday', '_hour', '_weekday_hour',
            '_amount_qcut10', 'Operating_system', 'Browser_type', 'DeviceType',
            'DeviceInfo', '_hour_density', '_amount_decimal', '_amount_decimal_len',
            '_amount_fraction', 'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
            'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3',
            '_hour_bucket_evening', '_hour_bucket_morning', '_hour_bucket_night',
            '_is_weekend_1'
        ]
        
        input_data = {}
        for field in input_fields:
            val = request.form.get(field, '0')
            try:
                input_data[field] = float(val)
            except:
                input_data[field] = val

        df = pd.DataFrame([input_data])

        # Cross features
        df['_P_emaildomain__addr1'] = df['P_emaildomain'].astype(str) + '__' + df['addr1'].astype(str)
        df['_card_id__issuer'] = df['card_id'].astype(str) + '__' + df['issuer_bank_code'].astype(str)
        df['_card_id__addr1'] = df['card_id'].astype(str) + '__' + df['addr1'].astype(str)
        df['_issuer__addr1'] = df['issuer_bank_code'].astype(str) + '__' + df['addr1'].astype(str)
        df['_cardid_issuer__addr1'] = df['_card_id__issuer'] + '__' + df['addr1'].astype(str)

        # Label encoding
        for col in label_encode_cols:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        pred = model.predict_proba(df)[:, 1][0]

        return render_template("index.html", prediction_text=f"Fraud Probability: {pred:.4f}")
    except Exception as e:
        return jsonify({'error': str(e)})
      

if __name__ == "__main__":
    app.run(debug=True)
