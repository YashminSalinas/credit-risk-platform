# 🚀 Credit Risk Scorecard Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)
![PyArrow](https://img.shields.io/badge/PyArrow-Parquet-4B8BBE?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git)
![VS Code](https://img.shields.io/badge/VS_Code-IDE-007ACC?style=for-the-badge&logo=visualstudiocode)
![Status](https://img.shields.io/badge/Status-Production_Style-success?style=for-the-badge)

</p>

| Dataset | Records | Outputs | Domain |
|---------|---------:|---------|--------|
| Lending Club | 2.2M+ | Credit Score · PD · Risk Bands | Credit Risk |

> End-to-end credit risk scorecard pipeline that transforms raw lending data
> into explainable credit scores, Probability of Default (PD), and risk bands
> using industry-standard credit risk modeling practices commonly adopted by financial institutions.
---
# 🏦 Business Problem

Financial institutions continuously face questions such as:

- Which borrowers are more likely to default?
- Which customer characteristics explain credit risk?
- How can historical lending data be transformed into predictive variables?
- How can interpretable scorecards be developed following industry standards?
- How can the entire modeling workflow be automated and reproduced?

This project addresses these challenges using the Lending Club public loan dataset and industry-standard credit risk modeling techniques.

---
## 💼 Business Value

This platform provides an end-to-end, explainable, and scalable credit risk assessment workflow that extends beyond binary default prediction.

### Key Business Outcomes

- 📉 Reduce expected credit losses through more accurate default prediction.
- ⚡ Accelerate loan underwriting with automated credit scoring.
- 🎯 Improve customer segmentation using standardized risk bands.
- 🔍 Increase model transparency with Weight of Evidence (WOE) and Information Value (IV).
- 📊 Support regulatory-friendly and explainable credit decision processes.
- 💰 Enable risk-based pricing strategies.
- 📦 Scale efficiently to millions of loan records through an optimized ETL architecture.

---

## 📈 Business Impact

| Business Challenge | Solution |
|--------------------|----------|
| Manual credit evaluation | Automated credit scoring |
| Inconsistent lending decisions | Standardized scorecard methodology |
| Poor portfolio segmentation | Risk bands based on Probability of Default |
| Limited model interpretability | Fully interpretable WOE transformations |
| Large raw datasets | Efficient ETL pipeline using Parquet |
| Model monitoring | Population Stability Index (PSI) for drift detection |

---

## ⚙️ Technical Scope

The pipeline consists of the following stages:

- ETL pipeline (Bronze → Silver → Gold)
- Data cleaning and preprocessing
- Feature engineering
- Information Value (IV)
- Weight of Evidence (WOE)
- Optimal binning
- Logistic Regression
- Probability calibration
- Score scaling
- Risk band assignment
- Population Stability Index (PSI)

---

## 🏗️ Architecture

```text
Raw Loan Data
      │
      ▼
 Bronze Layer
      │
      ▼
 Silver Layer
      │
      ▼
 Gold Layer
      │
      ▼
 Feature Engineering
      │
      ▼
 Scorecard Model
      │
      ▼
 Model Monitoring
      │
      ▼
 Business Outputs
```

---

## 📊 Example Business Decision

| Borrower | Probability of Default | Credit Score | Risk Band | Decision |
|----------|------------------------|--------------|-----------|----------|
| A | 8% | 742 | Low | Approve |
| B | 21% | 665 | Medium | Manual Review |
| C | 49% | 545 | High | Decline |

---

## 📂 Dataset

- **Source:** Lending Club public loan dataset
- **Records:** ~2.2 million loans
- **Features:** ~150 variables
- **Storage Format:** Parquet
- **Domain:** Consumer Lending / Credit Risk

---

## 📦 Outputs

- Probability of Default (PD)
- Credit Score
- Risk Band Classification
- WOE Transformation Tables
- Information Value Reports
- Engineered Feature Dataset
- Serialized Model Artifacts
- Population Stability Index (PSI) Metrics

---

## 💡 Why This Project Matters

Traditional machine learning projects often focus primarily on model training.

This project extends beyond model training by reproducing the complete lifecycle of a real-world banking credit scorecard, integrating:

- Data Engineering
- Credit Risk Analytics
- Explainable Machine Learning
- Model Monitoring
- Production-Oriented Architecture

The result is an end-to-end credit risk platform designed to resemble workflows commonly used in financial institutions.

---
