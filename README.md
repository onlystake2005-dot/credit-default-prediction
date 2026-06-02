# Credit Scoring Model

> **Internship Project** — Predicting individual creditworthiness using machine learning classification on financial data.

---

##  Project Overview

This project builds a credit risk assessment model that predicts the probability of loan default based on an applicant's financial history. It demonstrates the full machine learning workflow: data loading, feature engineering, model training, evaluation, and visualization.

**Objective:** Classify applicants as "Default" or "Non-Default" using past financial behavior.

---

##  Files

| File | Description |
|------|-------------|
| `credit_score_model.py` | Main script — run this |
| `credit_data_sample.csv` | Dataset (10,000 applicants, 23 features) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

##  Quick Start

```bash
pip install -r requirements.txt
python credit_score_model.py
```

The script will:
1. Load the dataset from `credit_data_sample.csv`
2. Engineer 11 composite features
3. Train 4 classification algorithms
4. Evaluate with cross-validation and holdout testing
5. Generate evaluation charts

---

##  Dataset Features

### Raw Features (12)
| Feature | Description |
|---------|-------------|
| `age` | Applicant age (18–75) |
| `annual_income` | Annual income in USD |
| `employment_years` | Years at current employment |
| `debt_to_income` | Debt-to-income ratio |
| `credit_history_length` | Years of credit history |
| `num_credit_accounts` | Number of open credit accounts |
| `num_delinquencies_2y` | Delinquencies in past 2 years |
| `credit_utilization` | Credit card utilization ratio |
| `num_hard_inquiries_6m` | Hard credit inquiries (6 months) |
| `has_mortgage` | Binary mortgage flag |
| `has_auto_loan` | Binary auto loan flag |
| `education_level` | 1=HS, 2=Some College, 3=Bachelor, 4=Graduate |

### Engineered Features (11)
| Feature | Description |
|---------|-------------|
| `debt_stress_score` | Composite risk score (DTI + Utilization + Delinquencies) |
| `financial_stability` | Stability index (Income + Employment + History + Education) |
| `dti_utilization_interaction` | Interaction between DTI and credit utilization |
| `age_income_interaction` | Age × log(Income) maturity proxy |
| `delinquency_rate` | Delinquencies per year of credit history |
| `inquiry_intensity` | Hard inquiries relative to history length |
| `income_per_account` | Income divided by number of accounts |
| `income_to_debt_ratio` | Income relative to estimated debt burden |
| `credit_maturity` | History length × number of accounts |
| `income_tier` | Income quintile (1–5) |
| `age_group` | Age bucket (1–5) |

---

##  Algorithms

| Algorithm | Notes |
|-----------|-------|
| **Logistic Regression** | Interpretable baseline; scaled features |
| **Decision Tree** | Non-linear splits; easy to explain |
| **Random Forest** | Ensemble of trees; handles interactions |
| **Gradient Boosting** | Sequential error correction |

Class imbalance (~4.5% default rate) is handled via `class_weight='balanced'`.

---

##  Evaluation

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Discrimination power across all thresholds |
| **Precision** | Of predicted defaults, how many actually defaulted? |
| **Recall** | Of actual defaults, how many did we catch? |
| **F1-Score** | Harmonic mean of precision and recall |

**Validation:** 5-Fold Stratified Cross-Validation + 25% Holdout Test Set

---

##  Results

| Model | Test ROC-AUC | Precision | Recall | F1-Score |
|-------|-------------|-----------|--------|----------|
| **Logistic Regression** | **0.7467** | 0.0888 | **0.6216** | 0.1554 |
| Random Forest | 0.7245 | **0.1901** | 0.2072 | **0.1983** |
| Gradient Boosting | 0.6973 | 0.1579 | 0.0270 | 0.0462 |
| Decision Tree | 0.6185 | 0.0624 | 0.4505 | 0.1096 |

**Best Model:** Logistic Regression (highest ROC-AUC and best recall for catching defaults)

---

##  Key Insights

- **Top Risk Predictors:** `debt_stress_score`, `dti_utilization_interaction`, `num_delinquencies_2y`
- **Protective Factors:** Higher `income_tier`, `age_income_interaction`, `education_level`
- **Class Imbalance:** 4.45% default rate is realistic but makes precision challenging
- **Calibration:** Logistic Regression provides well-calibrated probability estimates suitable for credit scorecards

---

##  Recommendations

1. **Threshold Tuning:** Adjust cutoff based on cost of missed defaults vs. lost good customers
2. **Ensemble:** Combine Logistic Regression (recall) + Random Forest (precision) via stacking
3. **Fairness:** Audit for disparate impact across demographic groups before deployment
4. **Monitoring:** Track Population Stability Index (PSI) and retrain quarterly
5. **Macro Features:** Add unemployment rate, interest rate environment, regional indicators

---

##  Tech Stack

- **Python 3.8+**
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **Scikit-learn** — ML algorithms & metrics
- **Matplotlib** — Visualization

---

*Generated for internship submission — June 2026*
