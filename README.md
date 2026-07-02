# Credit Risk Platform

## 🎯 Objective

The goal of this project is to design and implement an end-to-end credit risk platform using engineering and analytics practices commonly adopted in the financial industry.

The project covers the complete analytical workflow, from raw loan data ingestion to feature engineering and predictive modeling, following a Medallion Architecture (Bronze → Silver → Gold).

---

## 🏦 Business Problem

Financial institutions must answer critical questions such as:

- Which borrowers are more likely to default?
- Which financial characteristics increase credit risk?
- How can historical loan information be transformed into predictive features?
- How can large-scale credit data pipelines be built efficiently?

This project addresses these questions using the **Lending Club Loan Dataset**, following engineering practices commonly found in banking environments.

---

## 🧱 End-to-End Architecture

```

Raw Data (CSV)
↓
Bronze Layer
(Chunk Ingestion + Parquet)
↓
Silver Layer
(Data Cleaning + Target Engineering)
↓
Gold Layer
(Feature Engineering)
↓
Analytics Layer
(EDA + Visualization)
↓
Machine Learning
(Credit Risk Models)
↓
Model Monitoring
(Coming Soon)

```

---

## 📁 Project Structure

```

credit-risk-platform/
│
├── data/
│   ├── landing/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── model/
│
├── src/
│   ├── ingestion/
│   ├── analytics/
│   ├── modeling/
│   ├── monitoring/
│   ├── common/
│   ├── config/
│   └── etl/
│
├── reports/
├── logs/
├── notebooks/
├── sql/
│
├── requirements.txt
├── main.py
└── README.md

```

---

## 📊 Dataset

- **Source:** Lending Club Loan Dataset
- **Records processed:** ~2.26 million loans
- **Raw variables:** 151
- **Storage:** CSV → Parquet
- **Architecture:** Bronze → Silver → Gold

---

## ⚙️ Pipeline Breakdown

### 🚀 Bronze Layer (Data Ingestion)

- Chunk-based ingestion for multi-million row datasets
- Memory-efficient processing
- CSV to Parquet conversion
- Idempotent ingestion process
- Centralized data loading utilities

---

### 🧹 Silver Layer (Data Preparation)

- Binary target engineering
- Missing value reduction
- Numeric type standardization
- Analytical dataset preparation

---

### 🏅 Gold Layer (Feature Engineering)

Business-oriented features:

- Monthly income
- Income-to-loan ratio
- High DTI flag
- High revolving utilization flag
- FICO score buckets

---

### 📈 Analytics Layer

- Data Quality Assessment
- Exploratory Data Analysis
- Risk Visualization
- Business Insights

---

## 📊 Key Risk Insights

The analysis identified several important credit risk patterns:

- Lower FICO scores are associated with higher default rates.
- Defaulted borrowers tend to receive higher interest rates.
- Lower annual income is correlated with increased default probability.
- Higher Debt-to-Income (DTI) ratios indicate greater credit risk.

---

## 🛠 Tech Stack

### Data Engineering

- Python
- Pandas
- PyArrow
- Logging
- Pathlib

### Analytics

- NumPy
- Matplotlib

### Machine Learning

- Scikit-learn *(upcoming)*

### Big Data

- PySpark *(planned)*

### Dev Tools

- Git
- GitHub
- Visual Studio Code

---

## 🧠 Engineering Highlights

- Medallion Architecture (Bronze → Silver → Gold)
- Chunk-based ingestion
- Centralized data loading
- Modular pipeline design
- Production-style logging
- Memory-efficient processing
- Robust validation pipeline
- Analytics-ready datasets

---

# 🚀 Project Roadmap

## ✅ Sprint 1 — Data Engineering

- [x] Project structure
- [x] Large-scale CSV ingestion
- [x] Bronze Layer
- [x] Chunk processing
- [x] Parquet optimization
- [x] Logging

---

## ✅ Sprint 2 — Data Analytics

- [x] Data quality assessment
- [x] Target engineering
- [x] Exploratory Data Analysis
- [x] Risk visualization

---

## ✅ Sprint 3 — Feature Engineering

- [x] Silver Layer
- [x] Gold Layer
- [x] Business feature engineering

---

## ⏳ Sprint 4 — Predictive Modeling

- [ ] Feature selection
- [ ] Data leakage prevention
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Model evaluation
- [ ] Credit scoring

---

## ⏳ Sprint 5 — Model Explainability

- [ ] Feature importance
- [ ] SHAP explainability
- [ ] Risk interpretation
- [ ] Model comparison

---

## ⏳ Sprint 6 — Big Data Migration

- [ ] PySpark fundamentals
- [ ] Migrate ingestion pipeline to PySpark
- [ ] Spark DataFrames
- [ ] Spark SQL transformations
- [ ] Distributed Feature Engineering
- [ ] Performance benchmarking (Pandas vs Spark)

---

## ⏳ Sprint 7 — Pipeline Orchestration

- [ ] Pipeline modularization
- [ ] Docker
- [ ] Airflow
- [ ] Automated pipeline execution

---

## ⏳ Sprint 8 — Cloud Data Platform

- [ ] AWS S3
- [ ] AWS Glue
- [ ] Athena
- [ ] Cloud data lake
- [ ] Production-ready architecture

---

## 📌 Current Status

✔ Large-scale ingestion pipeline

✔ Bronze Layer

✔ Silver Layer

✔ Gold Layer

✔ Data Quality Assessment

✔ Target Engineering

✔ Exploratory Data Analysis

✔ Risk Visualization

✔ Feature Engineering

⏳ Machine Learning

⏳ Explainability

⏳ Spark Migration

⏳ Cloud Deployment

---

## 🎯 Long-Term Goals

- Production-ready Credit Risk Platform
- PySpark-based ETL
- Machine Learning credit scoring
- Cloud-native data platform
- Automated orchestration
- Model monitoring
- Explainable AI

---

## 👩‍💻 Project Type

This project is simulating a **real-world Credit Risk Platform** used by banking and financial institutions, combining **Data Engineering, Analytics, Machine Learning, Big Data and Cloud technologies**.