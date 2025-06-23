# Online Fraud Detection

This project aims to detect fraudulent transactions using the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection). A wide range of feature engineering techniques are applied, and a LightGBM model is used for classification. The final solution focuses on effective preprocessing, deep feature synthesis, and robust model training.

---

## 📅 Dataset Overview

The dataset combines transactional and identity information:

- `train_transaction.csv`: Contains transaction-specific features and the target variable `isFraud`.
- `train_identity.csv`: Includes identity and device-related fields.

The files are merged on `TransactionID`.

---

## 🌟 Objective

To build a supervised classification model that accurately predicts whether a transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`).

---

## 🔄 Target Column

| Column    | Description                     |
|-----------|---------------------------------|
| `isFraud` | 1 if the transaction is fraud, 0 if not |

---

## 📊 Feature Overview

### 💳 Transaction Details
- `TransactionAmt`: Amount of the transaction
- `ProductCD`: Product code
- `_amount_qcut10`: Amount binned into 10 quantiles
- `amount_decimal`, `amount_decimal_len`, `amount_fraction`: Engineered amount-based features

### 🗓️ Temporal Features
- `TransactionDT`: Relative timestamp
- `weekday`, `hour`, `weekday_hour`: Extracted from `TransactionDT`
- `hour_bucket`: Time of day (morning/afternoon/evening/night)
- `is_weekend`: Boolean flag for weekends
- `hour_density`: Transaction volume at each hour

### ⏳ Rolling & Time Gap Features
- `days_since_prev_txn`, `days_since_first_txn`
- `device_session_txn_gap`, `txn_gap_same_card`, `txn_gap_same_billing_addr`
- `rolling_txn_count_short_term`, `rolling_txn_count_mid_term`, `rolling_txn_count_long_term`
- `rolling_txn_time_short_term`, etc.

### 🆔 Identity & Card Details
- `card_id`, `issuer_bank_code`, `card_network`, `card_bin`, `card_type`
- `addr1`, `addr2`, `dist1`, `dist2`
- `match_status`: Engineered from `M4` (match/no match)
- `Operating_system`, `Browser_type`, `DeviceType`, `DeviceInfo`

### 🧑‍💻 Email & Address Cross Features
- `P_emaildomain`, `R_emaildomain`
- Cross: `P_emaildomain_addr1`, `card_id_issuer`, `card_id_addr1`, etc.
- Frequencies: `*_freq` for counts of these combinations

### 🧠 Behavioral Features
- `recent_txn_count`, `card_usage_frequency`, `shared_device_count`
- `billing_address_usage`, `shipping_address_usage`
- `transaction_type_count`, `device_browser_combo_count`
- `device_usage_frequency`, `inactive_device_count`
- `merchant_category_count`, `location_terminal_count`

---

## 🚀 Feature Engineering Techniques

### 1. **Time Extraction**
- `TransactionDT` converted to `weekday`, `hour`, `hour_bucket`
- `hour_density`: Used for behavior modeling

### 2. **Amount Features**
- `_amount_qcut10`: Binning transaction amounts into 10 deciles
- `amount_decimal`: Decimal portion of amount
- `amount_fraction`: Normalized decimal value

### 3. **Card and Address Crosses**
- Combining multiple categorical columns to capture interaction signals
- Frequency encoding applied to capture rarity/familiarity

### 4. **Rolling Windows**
- Count and time-based windows (short, mid, long) for transaction frequency and behavior analysis

---

## 🧠 Model - LightGBM

A LightGBM classifier is used for its speed, accuracy, and ability to handle large datasets and missing values.

### ⚖️ Parameters:
```python
LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    num_leaves=256,
    learning_rate=0.01,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    max_depth=-1,
    n_estimators=10000,
    early_stopping_rounds=100
)
