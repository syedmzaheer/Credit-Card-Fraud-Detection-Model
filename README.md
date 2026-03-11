# 💳 Credit Card Fraud Detection

A machine learning pipeline that trains and evaluates three classification models — **Logistic Regression**, **Decision Tree**, and **Random Forest** — to detect fraudulent credit card transactions using real-world transaction data.

---

## 📁 Project Structure

```
fraud-detection/
├── fraud_detection.py            # Main pipeline script
├── fraudTrain.csv                # Training dataset (you provide)
├── fraudTest.csv                 # Testing dataset (you provide)
├── fraud_detection_dashboard.png # Output: visual dashboard (auto-generated)
└── README.md                     # This file
```

---

## 📋 Requirements

### Python Version
- Python **3.8 or higher**

### Dependencies

Install all required packages with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Numerical computations |
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `matplotlib` | ≥ 3.4 | Plotting and dashboard |
| `seaborn` | ≥ 0.11 | Heatmaps for confusion matrices |
| `scikit-learn` | ≥ 1.0 | ML models, scaling, metrics |
| `imbalanced-learn` | ≥ 0.9 | SMOTE oversampling |

---

## 📊 Dataset

### Expected Files
The script expects **two separate CSV files** — one for training, one for testing. No train/test splitting is performed inside the script.

| File | Purpose |
|---|---|
| `fraudTrain.csv` | Used to train all models |
| `fraudTest.csv` | Used to evaluate all models |

### Required Columns

Both CSV files must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `trans_date_trans_time` | datetime | Timestamp of the transaction |
| `cc_num` | string/int | Credit card number |
| `merchant` | string | Merchant name |
| `category` | string | Merchant category (e.g. `grocery_pos`, `gas_transport`) |
| `amt` | float | Transaction amount in USD |
| `first` | string | Cardholder first name |
| `last` | string | Cardholder last name |
| `gender` | string | Cardholder gender (`M` / `F`) |
| `street` | string | Cardholder street address |
| `city` | string | Cardholder city |
| `state` | string | Cardholder state |
| `zip` | int | Cardholder ZIP code |
| `lat` | float | Cardholder latitude |
| `long` | float | Cardholder longitude |
| `city_pop` | int | Population of cardholder's city |
| `job` | string | Cardholder's occupation |
| `dob` | date | Cardholder date of birth |
| `trans_num` | string | Unique transaction ID |
| `unix_time` | int | Unix timestamp of transaction |
| `merch_lat` | float | Merchant latitude |
| `merch_long` | float | Merchant longitude |
| `is_fraud` | int (0/1) | **Target label** — `1` = fraud, `0` = legitimate |

---

## 🚀 How to Run

**Step 1** — Clone or download the project files.

**Step 2** — Place your dataset files in the same folder as the script:
```
fraud_detection.py
fraudTrain.csv
fraudTest.csv
```

**Step 3** — Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

**Step 4** — Run the script:
```bash
python fraud_detection.py
```

**Step 5** — Check the output:
- Classification reports printed to the console
- `fraud_detection_dashboard.png` saved in the same folder

---

## ⚙️ Pipeline Steps

The script executes the following steps in order:

### 1. Load Datasets
Reads `fraudTrain.csv` and `fraudTest.csv` directly — no splitting is done.

### 2. Feature Engineering
Derives new features from the raw columns:

| New Feature | Source | Description |
|---|---|---|
| `log_amt` | `amt` | Log-transformed transaction amount to reduce skew |
| `hour` | `trans_date_trans_time` | Hour of day (0–23) |
| `day_of_week` | `trans_date_trans_time` | Day of week (0=Mon, 6=Sun) |
| `month` | `trans_date_trans_time` | Month of year (1–12) |
| `is_night` | `hour` | 1 if transaction is between 10 PM – 6 AM |
| `age` | `dob` | Cardholder age at time of transaction |
| `distance_km` | `lat`, `long`, `merch_lat`, `merch_long` | Haversine distance between cardholder home and merchant |
| `gender_enc` | `gender` | Binary encoded gender (1=Male, 0=Female) |
| `category_enc` | `category` | Label-encoded merchant category |
| `log_city_pop` | `city_pop` | Log-transformed city population |
| `zip` | `zip` | ZIP code (used as a regional signal) |

### 3. Target Cleaning
Rows where `is_fraud` is `NaN` are dropped from both train and test sets. The column is cast to `int`.

### 4. Feature Imputation
Any `NaN` or `Inf` values in the feature matrix are replaced using **column medians computed from the training set only** — preventing data leakage into the test set.

### 5. Feature Scaling
All features are standardized using `StandardScaler`. The scaler is **fit on the training set only** and then applied to both train and test.

### 6. SMOTE Oversampling
Since fraud cases are a small minority of transactions, **SMOTE (Synthetic Minority Oversampling Technique)** is applied to the training set to balance the classes. SMOTE is **never applied to the test set**.

### 7. Model Training
Three models are trained on the SMOTE-balanced training data:

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `C=0.1`, `max_iter=1000` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=10` |
| Random Forest | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=5` |

### 8. Evaluation
Each model is evaluated on the original (unbalanced) test set using:
- **ROC-AUC** — Area under the ROC curve
- **PR-AUC** — Area under the Precision-Recall curve (best metric for imbalanced data)
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — True positives, false positives, false negatives, true negatives

### 9. Dashboard
A visual dashboard is saved as `fraud_detection_dashboard.png` containing:
- Class distribution bar chart
- Random Forest feature importances
- Log(amount) histogram by class
- Confusion matrix for each model
- ROC curves for all models
- Precision-Recall curves for all models
- ROC-AUC and F1-Score comparison bar charts
- Performance summary table

---

## 📈 Output Example

**Console output:**
```
=================================================================
  CREDIT CARD FRAUD DETECTION — REAL DATASET PIPELINE
=================================================================

📂 Loading datasets ...
   fraudTrain.csv  →  1,048,575 rows
   fraudTest.csv   →    555,719 rows

🔧 Engineering features ...
🧹 Dropping rows with NaN in is_fraud target ...
🧹 Checking for NaN / Inf values ...
   Imputation complete ✔

✅ Train : 1,048,575 rows  |  Fraud = 7,506  (0.72%)
✅ Test  : 555,719 rows   |  Fraud = 2,145  (0.39%)

⚖️  Applying SMOTE to training set ...
   After SMOTE → 1,041,069 legitimate  /  1,041,069 fraud

🤖 Training on fraudTrain.csv ...

   [Random Forest] ROC-AUC=0.9821  PR-AUC=0.8934  F1=0.8102
   ...

🏆  Best model: Random Forest  (ROC-AUC = 0.9821)
```

---

## 🔑 Key Design Decisions

**No data leakage** — The scaler, medians, and SMOTE are all fit exclusively on training data and applied to the test set, ensuring a fair evaluation.

**SMOTE on train only** — Oversampling is applied only to the training set. The test set remains in its original distribution to reflect real-world conditions.

**PR-AUC over accuracy** — With heavily imbalanced fraud data (~1% fraud rate), accuracy is misleading. PR-AUC is the primary metric as it reflects performance on the minority class.

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: fraudTrain.csv` | CSV not in same folder as script | Move both CSVs next to `fraud_detection.py` |
| `ValueError: Input y contains NaN` | NaN values in `is_fraud` column | Already handled — script drops these rows automatically |
| `ValueError: Input X contains NaN` | Missing values in feature columns | Already handled — imputed with train medians |
| `KeyError: 'is_fraud'` | Column named differently in your CSV | Rename the target column to `is_fraud` |
| `ModuleNotFoundError` | Missing dependency | Run `pip install imbalanced-learn scikit-learn` |

---

## 📄 License

This project is for educational and research purposes.
