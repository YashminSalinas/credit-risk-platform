# Credit Risk Analytics Platform (Banking Simulation Project)

## 🎯 Objective

This project simulates a real-world **credit risk analytics pipeline** used in financial institutions to evaluate borrower default risk.

It replicates how data flows in a risk department: from raw loan data ingestion to analytical insights and modeling preparation.

---

## 🏦 Business Problem

Financial institutions must assess:

- Probability of borrower default
- Financial behavior patterns of applicants
- Risk drivers behind credit decisions

This project addresses these questions using historical Lending Club loan data.

---

## 🧱 End-to-End Architecture
Raw Data (Lending Club)
↓
Data Ingestion Layer (Chunk processing + Parquet optimization)
↓
Analytical Layer (Target engineering + EDA)
↓
Visualization Layer (Risk insights + validation)
↓
Future: Feature Engineering → ML Model → Credit Scoring System

---

credit-risk-platform/
│
├── data/
│   ├── landing/              # raw files (optional future layer)
│   ├── bronze/               # processed raw data (Parquet chunks)
│   ├── silver/               # cleaned data (future)
│   ├── gold/                 # analytics-ready data (future)
│
├── src/
│   ├── ingestion/            # data ingestion pipeline
│   │   └── load_raw_data.py
│   │
│   ├── analytics/            # EDA + risk analysis
│   │   ├── define_target.py
│   │   ├── eda_risk.py
│   │   └── visual_risk.py
│   │
│   ├── etl/                  # transformations (future)
│   ├── modeling/             # ML models (future)
│   ├── monitoring/           # monitoring (future)
│   └── common/               # shared utilities (future)
│
├── reports/
│   └── figures/              # generated EDA visualizations
│
├── logs/                     # execution logs per run
│
├── notebooks/                # exploratory notebooks (optional)
├── sql/                     # future analytics queries
│
├── requirements.txt
├── main.py
└── README.md

---

## ⚙️ Pipeline Breakdown

### 🚀 Data Ingestion Layer
- Processed large-scale CSV datasets
- Implemented chunk-based ingestion to handle memory constraints
- Stored structured data in Parquet format (Bronze layer)
- Ensured scalable and efficient data handling

---

### 📊 Target Engineering (Credit Definition)
- Defined binary risk target:
  - `0 → PAGADOR (No default)`
  - `1 → MOROSO (Default)`
- Standardized inconsistent loan status categories
- Built clean label for predictive modeling

---

### 📈 Exploratory Data Analysis (EDA)
Analyzed key risk variables:

- Credit score (FICO)
- Interest rate
- Annual income
- Debt-to-income ratio (DTI)

Identified clear risk separation between default and non-default borrowers.

---

### 📉 Visualization & Data Quality Layer
- Built robust visualization pipeline (histograms + boxplots)
- Implemented data validation before plotting
- Handled missing values and inconsistent types
- Prevented visualization failures in production-like conditions

---

## 📊 Key Risk Insights

- Lower FICO scores are strongly associated with default risk
- Default borrowers show higher interest rates
- Income levels are significantly lower in default population
- Higher DTI indicates higher probability of default

---

## 🛠 Tech Stack

- Python (Data Engineering + Analytics)
- Pandas (ETL & transformation)
- Matplotlib (EDA visualization)
- PyArrow (Parquet storage)
- Git & GitHub (version control)
- VSCode (development)

---

## 🧠 Engineering Highlights

- Chunk-based ingestion for large datasets
- Data validation layer before visualization
- Modular pipeline design (ingestion → analytics → visualization)
- Production-style logging and run tracking
- Robust handling of missing and inconsistent data

---

## 📌 Current Status

✔ Data ingestion pipeline  
✔ Target engineering  
✔ EDA analysis completed  
✔ Visualization layer stabilized  
⏳ Feature engineering (next phase)  
⏳ Predictive modeling (credit scoring)  

---

## 🚀 Next Steps

- Feature engineering (risk ratios, behavioral features)
- Credit scoring model (logistic regression baseline)
- Model evaluation (AUC, KS, confusion matrix)
- Scorecard interpretability layer

---

## 👩‍💻 Project Type

Portfolio project simulating **credit risk analytics in a banking environment**