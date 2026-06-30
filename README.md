# Predicting Carryover Risk Among Polytechnic Students Using Supervised Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Methodology-CRISP--DM-orange.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![XAI](https://img.shields.io/badge/Explainable%20AI-SHAP-green.svg)](https://github.com/shap/shap)

A comprehensive final-year undergraduate and Higher National Diploma (HND) project framework for applying supervised machine learning classifiers to predict student academic risk in practical science courses. This system builds an early-warning pipeline using institutional performance tracking data from course **STP 213 (Practical Science)**.

---

## 📌 Project Architecture & Overview
* **Course:** STP 213 — Practical Science
* **Session:** 2025/2026
* **Academic Level:** HND / ND Final Year
* **Supervisor:** OLUMODEJI I.A.


### Abstract
Traditional academic management structures rely on reactive student support mechanism]. This research framework implements a proactive machine learning pipeline designed to flag at-risk students before the semester concludes. Benchmarking six supervised algorithms—Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbours (KNN), and XGBoost—the core engine addresses heavy class imbalances using SMOTE and incorporates model interpretability using SHAP values.

---

## 🛠️ Data Mining Pipeline (CRISP-DM)
The workflow follows the Cross-Industry Standard Process for Data Mining:
1. **Data Cleaning:** Handles non-participation anomalies by isolating totally absent students (scores of zero across all fields) and maps them into the highest risk class.
2. **Feature Engineering:** Computes normalized ratio matrices and cross-component score differences to extract maximum signal from performance records.
3. **Class Balancing (SMOTE):** Dynamically synthesizes minority class profiles within the training partition to eliminate algorithmic bias toward passing records.
4. **Explainable AI Integration:** Deploys TreeExplainer SHAP modules to identify global feature impact and individual risk indicators.

---

## 📊 Core Engineered Feature Matrix

| Feature Name | Derived Formula / Mapping | Rationale |
| :--- | :--- | :--- |
| `Exam_ratio` | $\text{Exam} / 40$ | Normalizes raw examination score to a $0–1$ scale. |
| `Pract_ratio` | $\text{Pract} / 40$ | Normalizes practical performance to a $0–1$ scale. |
| `CA_ratio` | $\text{CA} / 20$ | Normalizes continuous assessment to a $0–1$ scale. |
| `Absent` | If Exam, Pract, & CA == 0 $\rightarrow 1$ else $0$ | Isolates early-semester chronic non-participation. |
| `Pass_margin` | $\text{Total} - 40$ | Evaluates proximity metric from the official pass boundary. |

---

## 📈 Model Performance Benchmark Template

Evaluation metrics are computed on the unbalanced test split to ensure historical integrity, prioritizing **Recall (Sensitivity)** to eliminate missed at-risk flags[cite: 1].

| Model Classifier | Accuracy | Precision | Recall (Sensitivity) | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | *0.98* | *1.0* | *0.8889* | *0.9412* | *0.9458* |
| **Decision Tree** | *0.98* | *1.0* | *0.8889* | *0.9412* | *0.9986* |
| **Random Forest** | *1.00* | *1.0* | *1.0000* | *1.0000* | *1.0000* |
| **SVM** | *0.98* | *1.0* | *0.8889* | *0.9412* | *1.0000* |
| **KNN** | *0.98* | *1.0* | *0.8889* | *0.9412* | *1.0000* |
| **XGBoost** | *0.98* | *1.0* | *0.8889* | *0.9412* | *1.0000* |

---

## 📂 Project Repository Directory
```text
├── data/                      # Raw and anonymized evaluation sheets (.xlsx)
├── notebooks/                 # Exploratory Data Analysis & visual plots
├── src/
│   └── pipeline.py            # Complete end-to-end Python engine code
├── figures/                   # Confusion matrices, ROC curves, and SHAP outputs
├── requirements.txt           # Environment dependencies list
└── README.md                  # System documentation summary
