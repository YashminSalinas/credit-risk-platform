# Credit Risk Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)

![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)

![PyArrow](https://img.shields.io/badge/PyArrow-Parquet-4B8BBE?style=for-the-badge)

![Git](https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git)

![VS Code](https://img.shields.io/badge/VS_Code-IDE-007ACC?style=for-the-badge&logo=visualstudiocode)

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</p>

---

> **Production-Style Credit Risk Scorecard Platform**
>
> End-to-end machine learning pipeline implementing feature engineering,
> Information Value (IV), Weight of Evidence (WOE), optimal binning,
> probability calibration, score scaling and risk band assignment
> following credit risk modeling practices commonly used in banking
> and financial institutions.

---

# 🎯 Objective

This project implements an end-to-end Credit Risk Scorecard Platform inspired by analytical and engineering workflows commonly adopted in banking and financial institutions.

The platform covers the complete lifecycle of a credit risk model, from raw loan data ingestion to scorecard development, probability calibration, risk segmentation, and score generation using a modular and production-oriented architecture.

---

# 🏦 Business Problem

Financial institutions need to answer questions such as:

- Which borrowers are more likely to default?
- Which customer characteristics explain credit risk?
- How can historical lending data be transformed into predictive variables?
- How can interpretable scorecards be developed following industry standards?
- How can the entire modeling workflow be automated and reproduced?

This project addresses these questions using the Lending Club Loan Dataset while implementing techniques commonly used in real-world credit risk modeling.

---

# 🧱 End-to-End Architecture

```text
Raw Loan Data (CSV)
        │
        ▼
Bronze Layer
(Chunk Ingestion + Parquet)
        │
        ▼
Silver Layer
(Data Cleaning + Target Engineering)
        │
        ▼
Gold Layer
(Feature Engineering)
        │
        ▼
Feature Selection
        │
        ▼
Optimal Binning
        │
        ▼
Information Value Analysis
        │
        ▼
Rare Category Grouping
        │
        ▼
Monotonic WOE Transformation
        │
        ▼
Correlation Filtering
        │
        ▼
LogisticRegressionCV
        │
        ▼
Probability Calibration
        │
        ▼
Credit Score Scaling
        │
        ▼
Risk Band Assignment
        │
        ▼
Model Artifacts
```

---

# 📁 Project Structure

```text
credit-risk-platform/
│
├── data/
│   ├── landing/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── model/
│       └── datasets/
│
├── reports/
│
├── tests/
│
├── src/
│   │
│   ├── analytics/
│   │   ├── define_target.py
│   │   ├── eda_risk.py
│   │   └── visual_risk.py
│   │
│   ├── common/
│   │   └── io_utils.py
│   │
│   ├── ingestion/
│   │   └── load_raw_data.py
│   │
│   ├── modeling/
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py
│   │   ├── prepare_features.py
│   │   └── train_scorecard.py
│   │
│   ├── scorecard/
│   │   ├── config.py
│   │   ├── binning.py
│   │   ├── woe.py
│   │   ├── correlation.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── scoring.py
│   │   └── artifacts.py
│   │
│   ├── monitoring/
│   │
│   └── warehouse/
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

- Dataset: Lending Club Loan Dataset
- Original Records: ~2.26 Million Loans
- Original Features: 151
- Storage Format: CSV → Parquet
- Architecture: Medallion (Bronze → Silver → Gold)

---

# ⚙️ Pipeline Overview

## 🚀 Data Ingestion

- Chunk-based ingestion
- Memory-efficient processing
- CSV to Parquet conversion
- Idempotent loading
- Centralized data utilities

---

## 🧹 Data Preparation

- Target engineering
- Missing value handling
- Data type standardization
- Analytical dataset generation

---

## 🏅 Feature Engineering

Business-oriented variables including:

- Monthly income
- Income-to-loan ratio
- FICO buckets
- High DTI flag
- Revolving utilization flag
- Credit behavior indicators
- Derived financial attributes

---

## 📊 Exploratory Analytics

- Data Quality Assessment
- Missing Value Analysis
- Default Rate Analysis
- Risk Visualizations
- Business Insights

---

# 🧠 Credit Scorecard Development

The modeling pipeline implements techniques commonly adopted in financial institutions.

## Feature Selection

- Information Value (IV)
- IV stability validation
- Predictive feature filtering

---

## Optimal Binning

- Supervised binning
- Business-oriented discretization
- Rare category grouping

---

## WOE Transformation

- Weight of Evidence encoding
- Monotonicity validation
- Production-ready categorical mappings

---

## Model Validation

- Leakage prevention
- Correlation filtering
- Feature redundancy removal

---

## Model Training

- LogisticRegressionCV
- Cross-validation
- Automatic regularization tuning

---

## Probability Calibration

- CalibratedClassifierCV
- Sigmoid calibration
- Probability of Default estimation

---

## Credit Scoring

- Score scaling
- Risk band assignment
- Scorecard generation

---

# 📈 Model Outputs

The platform automatically generates:

- Selected features
- WOE mappings
- Information Value reports
- Feature selection reports
- Calibrated models
- Credit scores
- Risk bands
- Production-ready datasets

---

# 📊 Key Risk Insights

The analysis identifies several well-known credit risk relationships:

- Lower FICO scores are associated with higher default rates.
- Higher interest rates are correlated with increased default probability.
- Lower income generally implies higher credit risk.
- Higher Debt-to-Income ratios increase default likelihood.
- Credit utilization contributes to borrower risk segmentation.

---

# 🛠 Tech Stack

## Data Engineering

- Python
- Pandas
- NumPy
- PyArrow

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib

### Development

- Git
- GitHub
- Visual Studio Code

---

# ⭐ Engineering Highlights

- Medallion Architecture
- Modular Pipeline Design
- Chunk-Based Processing
- Memory-Efficient ETL
- Feature Engineering Pipeline
- Information Value Analysis
- IV Stability Validation
- Optimal Binning
- Rare Category Handling
- Monotonic WOE Transformation
- Correlation Filtering
- Leakage Prevention
- LogisticRegressionCV
- Probability Calibration
- Credit Score Scaling
- Automated Risk Band Assignment
- Production-Oriented Artifacts

---

# 📌 Current Status

✅ Large-scale data ingestion

✅ Medallion Architecture

✅ Feature Engineering

✅ Exploratory Data Analysis

✅ Data Quality Assessment

✅ Information Value Analysis

✅ Feature Selection

✅ Optimal Binning

✅ WOE Encoding

✅ Rare Category Grouping

✅ Monotonicity Validation

✅ Correlation Filtering

✅ LogisticRegressionCV

✅ Probability Calibration

✅ Credit Score Scaling

✅ Risk Band Assignment

✅ Model Artifact Generation

⏳ Model Monitoring

⏳ Population Stability Index (PSI)

⏳ Production Drift Monitoring

---

# 🎯 Long-Term Goals

- Production-ready Credit Risk Platform
- Automated Model Monitoring
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)
- Drift Detection
- Model Governance
- Production Deployment

---

# 👩‍💻 Project Type

This repository simulates a production-style Credit Risk Scorecard Platform inspired by workflows commonly implemented in banking and financial institutions.

The project combines **Data Engineering, Credit Risk Analytics, Feature Engineering, Statistical Modeling, Probability Calibration, and Credit Scorecard Development** using a modular and production-oriented architecture.