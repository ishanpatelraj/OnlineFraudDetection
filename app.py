from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import json
import os
import datetime

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

with open("amount_bins.pkl", "rb") as f:
    amount_bins = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # You must create templates/index.html

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Define fields based on image
        categorical_fields = [
            'ProductCD', 'card_network', 'card_type',
            'P_emaildomain', 'R_emaildomain',
            'DeviceType', 'Operating_system', 'Browser_type'
        ]

        numerical_fields = [
            'TransactionID', 'TransactionDT', 'TransactionAmt',
            'issuer_bank_code', 'card_bin',
            'addr1', 'addr2', 'dist1', 'dist2'
        ]

        additional_numericals = [
        'recent_txn_count', 'card_usage_frequency', 'shared_device_count',
        'billing_address_usage', 'shipping_address_usage', 'device_browser_combo_count',
        'transaction_type_count', 'device_usage_frequency', 'inactive_device_count',
        'merchant_category_count', 'location_terminal_count',
        'rolling_txn_count_short_term', 'rolling_txn_count_mid_term', 'rolling_txn_count_long_term'
        ]

        numerical_fields += additional_numericals

        string_fields = ['card_id', 'DeviceInfo']  # Treat as free-form strings

        input_fields = categorical_fields + numerical_fields + string_fields

        # Parse form data
        input_data = {}
        for field in input_fields:
            val = request.form.get(field, '0')
            if field in numerical_fields:
                try:
                    input_data[field] = float(val)
                except ValueError:
                    input_data[field] = 0.0
            else:
                input_data[field] = str(val)

        # Create DataFrame
        df = pd.DataFrame([input_data])

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

        # 🌟 Extract date-based features
        df['ymd'] = df['Date'].dt.year.astype(str) + '-' + df['Date'].dt.month.astype(str) + '-' + df['Date'].dt.day.astype(str)
        df['year_month'] = df['Date'].dt.year.astype(str) + '-' + df['Date'].dt.month.astype(str)
        df['weekday'] = df['Date'].dt.dayofweek
        df['hour'] = df['Date'].dt.hour
        df['day'] = df['Date'].dt.day
        df['_seq_day'] = df['TransactionDT'] // (24 * 60 * 60)
        df['_seq_week'] = df['_seq_day'] // 7
        df['_weekday_hour'] = df['weekday'].astype(str) + '_' + df['hour'].astype(str)
        df['_amount_qcut10'] = pd.cut(df['TransactionAmt'], bins=amount_bins, include_lowest=True)

        df['_weekday_hour'] = df['weekday'].astype(str) + '_' + df['hour'].astype(str)

        # _hour_bucket feature
        def get_time_bucket(hour):
            hour = int(hour)
            if 5 <= hour < 12: return 'morning'
            elif 12 <= hour < 17: return 'afternoon'
            elif 17 <= hour < 21: return 'evening'
            else: return 'night'

        df['_hour_bucket'] = df['hour'].apply(get_time_bucket)

        # _is_weekend feature
        df['_is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

        # _hour_density using fixed hour distribution (from training)
        hour_distribution = {
            0: 0.035, 1: 0.027, 2: 0.025, 3: 0.030, 4: 0.032,
            5: 0.040, 6: 0.045, 7: 0.050, 8: 0.060, 9: 0.065,
            10: 0.070, 11: 0.075, 12: 0.070, 13: 0.065, 14: 0.060,
            15: 0.058, 16: 0.055, 17: 0.052, 18: 0.050, 19: 0.045,
            20: 0.040, 21: 0.038, 22: 0.037, 23: 0.035
        }

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

        # Drop Date if not in model
        df.drop(columns=['Date'], inplace=True)

        d_features = [
            'days_since_prev_txn', 'days_since_first_txn', 'device_session_txn_gap',
            'txn_gap_same_card', 'txn_gap_same_billing_addr', 'days_since_last_login',
            'days_since_last_device_use', 'txn_gap_same_state', 'address_reuse_duration',
            'billing_shipping_time_diff', 'days_since_card_registration',
            'rolling_txn_time_short_term', 'rolling_txn_time_mid_term',
            'rolling_txn_time_long_term', 'rolling_txn_time_extended'
        ]
        for col in d_features:
            df[col] = 0

        import re

        df['_amount_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)

        df['_amount_decimal_len'] = df['TransactionAmt'].apply(
            lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]) if '.' in str(x) else 0
        )

        df['_amount_fraction'] = df['TransactionAmt'].apply(
            lambda x: float('0.' + re.sub(r'^[0-9]|\.|0+$', '', str(x))) if '.' in str(x) and re.search(r'\d', str(x).split('.')[-1]) else 0.0
        )

        # Label encoding for categorical fields
        for col in categorical_fields + string_fields:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Add any missing columns required by model
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        # Final cleanup
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        pred = model.predict_proba(df)[:, 1][0]

        return render_template("index.html", prediction_text=f"Fraud Probability: {pred:.4f}")
    
    except Exception as e:
        return jsonify({'error': str(e)})
      

if __name__ == "__main__":
    app.run(debug=True)
