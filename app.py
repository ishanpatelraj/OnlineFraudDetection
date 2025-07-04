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

        # Drop Date if not in model
        df.drop(columns=['Date'], inplace=True)

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
