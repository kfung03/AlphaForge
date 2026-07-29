# 📈 AlphaForge

> End-to-end quantitative machine learning platform for financial return prediction using
> Fama–French factor models, GARCH volatility estimation, path-signature feature engineering,
> deep learning, and portfolio backtesting.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview

AlphaForge is an end-to-end quantitative research platform that explores whether
advanced time-series representations improve financial return prediction.

Instead of relying solely on traditional asset-pricing models, this project combines:

- Fama–French 3- and 5-factor models
- GARCH volatility estimation
- Log-signature feature engineering
- CNN and XGBoost models
- Ensemble learning
- Portfolio backtesting

The final product includes a fully interactive Streamlit dashboard for exploring
the research workflow and model performance.

---

## Research Pipeline

```text
Raw Financial Data
        │
        ▼
Data Cleaning
        │
        ▼
Rolling Dataset Construction
        │
        ▼
Feature Engineering
        │
        ├── Fama-French Factors
        ├── GARCH Features
        └── Log-Signature Features
        │
        ▼
Machine Learning Models
        │
        ├── CNN
        ├── Random Forest
        ├── XGBoost
        └── Ensemble
        │
        ▼
Model Evaluation
        │
        ▼
Portfolio Construction
        │
        ▼
Robustness Analysis
```

---

# Features

✔ Financial data preprocessing

✔ Daily and monthly prediction pipelines

✔ Fama–French factor modeling

✔ GARCH volatility estimation

✔ Path-signature feature engineering

✔ CNN classifiers

✔ Random Forest

✔ XGBoost

✔ Ensemble learning

✔ Threshold optimization

✔ Portfolio backtesting

✔ Robustness analysis

✔ Interactive Streamlit dashboard

---

# Repository Structure

```text
alphaforge/

├── notebooks/
│
├── src/
│
├── data/
│
├── models/
│
├── results/
│
├── ui/
│
└── README.md
```

---

# Interactive Dashboard

The Streamlit application includes:

- Project Overview
- Data Pipeline
- Feature Engineering
- Model Evaluation
- Portfolio Backtesting
- About the Project

![Dashboard](images/dashboard.png)

---

# Models

The project compares multiple machine-learning approaches.

| Model | Purpose |
|--------|----------|
| CNN | Deep learning classifier |
| Random Forest | Tree ensemble baseline |
| XGBoost | Gradient boosted trees |
| Ensemble | Combines model predictions |

---

# Feature Engineering

Three feature families are compared.

| Feature Type | Description |
|-------------|-------------|
| Fama–French | Economic factor exposure |
| GARCH | Conditional volatility |
| Log-Signature | Sequential path representation |
| Combined | Feature fusion |

---

# Evaluation Metrics

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Portfolio

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

---

# Technologies

- Python
- pandas
- NumPy
- TensorFlow
- scikit-learn
- XGBoost
- ARCH
- Plotly
- Streamlit
- Git

---

# Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run ui/Home.py
```

---

# Future Improvements

- Transformer-based models
- Hyperparameter optimization
- Explainable AI (SHAP)
- Live market data integration
- Automated retraining pipeline

---

# Author

**Kyler Fung**

University of California, Davis

Finance • Machine Learning • Quantitative Research

GitHub: https://github.com/kfung03
LinkedIn: https://www.linkedin.com/in/kyler-fung/