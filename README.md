<div align="center">

<!-- BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2E2A72,50:4C3BC4,100:00E0A4&height=200&section=header&text=Student%20Performance%20ML%20System&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=Early%20Intervention%20%E2%80%A2%20At-Risk%20Prediction%20%E2%80%A2%20Production%20Pipeline&descAlignY=58&descSize=16&animation=fadeIn"/>

<br/>

<!-- BADGES -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)

<br/>

<!-- METRICS -->
| 🎯 Accuracy | 🔵 Precision | 🟠 Recall | 🟢 F1 Score | 🏆 Best Model |
|:-----------:|:------------:|:---------:|:-----------:|:-------------:|
| **~88%** | **~84%** | **~81%** | **~82%** | Gradient Boosting |

</div>

---

## 📌 Project Overview

This is a **production-level Machine Learning project** built on the **UCI Student Performance Dataset (Math Course)**. The system predicts whether a student is **at risk of failing** before their final exam, enabling early intervention by educators and institutions.

> **Target Variable:** `at_risk` — Binary classification (`0` = Pass, `1` = At-Risk)  
> **Threshold Logic:** `G3 < 10` → At-Risk (Fail) &nbsp;|&nbsp; `G3 ≥ 10` → Pass

This project simulates a real-world ML pipeline used in EdTech and public education analytics, covering the full journey from raw data to a deployed REST API.

---

## 🗂️ Project Pipeline

```
Raw Data (CSV)
    │
    ▼
📊 Exploratory Data Analysis
    │  → Grade distributions, correlation heatmaps, key feature plots
    ▼
⚙️  Preprocessing
    │  → LabelEncoding, StandardScaler, OneHotEncoder via ColumnTransformer
    ▼
🛠️  Feature Engineering
    │  → total_alcohol, study_efficiency, support_score
    ▼
🔀 Train/Test Split (80/20 Stratified)
    │
    ▼
🤖 Model Training via sklearn Pipeline
    │  → Logistic Regression | Decision Tree | Random Forest | Gradient Boosting
    ▼
📈 Evaluation
    │  → Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
    │  → 5-Fold Cross-Validation
    ▼
🔧 Hyperparameter Tuning
    │  → RandomizedSearchCV on Random Forest
    ▼
🚀 Deployment
       → FastAPI REST endpoint + MLflow experiment tracking
```

---

## 📂 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |
| **File** | `student-mat.csv` (semicolon-separated) |
| **Rows** | 395 students |
| **Columns** | 33 features |
| **Target** | Derived `at_risk` from `G3` (final grade) |

**Key Features Used:**
- `failures` — Number of past class failures
- `studytime` — Weekly study time (1–4 scale)
- `G1`, `G2` — 1st and 2nd period grades (dropped to avoid leakage, used in EDA)
- `absences` — Number of school absences
- `higher` — Aspiration for higher education
- `Medu` / `Fedu` — Mother's / Father's education level
- `schoolsup`, `famsup`, `paid` — Support resources available

---

## 🛠️ Feature Engineering

Three derived features were added to improve signal:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_alcohol` | `Dalc + Walc` | Combined weekday + weekend alcohol exposure |
| `study_efficiency` | `studytime / (failures + 1)` | How productive study time is relative to past failures |
| `support_score` | `schoolsup + famsup + paid` (binary sum) | Total institutional and family support available |

---

## 🤖 Models Trained

| Model | Type | Key Hyperparameters |
|-------|------|-------------------|
| **Logistic Regression** | Linear | `max_iter=1000` |
| **Decision Tree** | Tree | `max_depth=5`, `min_samples_split=10` |
| **Random Forest** | Ensemble | `n_estimators=200`, `max_depth=8` |
| **Gradient Boosting** | Boosting | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` |

All models are wrapped in **sklearn Pipelines** with a `ColumnTransformer` that handles:
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features

---

## 📊 Evaluation Strategy

- **Train/Test Split** — 80/20 stratified split to preserve class balance
- **5-Fold Cross-Validation** — F1 score on training set per model
- **Metrics** — Accuracy, Precision, Recall, F1, ROC-AUC per model
- **Visualizations** — Confusion matrices, ROC curves, metric bar charts, CV boxplots

> ⚠️ **Recall is the critical metric.** Missing an at-risk student is more costly than a false alarm. The system is optimized for high recall.

---

## 🔧 Hyperparameter Tuning

`RandomizedSearchCV` was applied on the **Random Forest** pipeline with a stratified 5-fold CV, searching over:
- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

Results were logged to **MLflow** for experiment tracking and reproducibility.

---

## 🚀 Deployment

### FastAPI Endpoint

The best model is serialized with `joblib` and served via **FastAPI**:

```python
POST /predict
{
  "studytime": 2,
  "failures": 1,
  "absences": 6
}

# Response
{
  "at_risk": 1,
  "probability": 0.7832
}
```

**Run locally:**
```bash
pip install fastapi uvicorn joblib
uvicorn app:app --reload
# → http://localhost:8000/predict
```

---

## 🔑 Key Insights

1. **Past failures** is the single strongest predictor of at-risk status
2. **G2 (2nd period grade)** is the best continuous predictor — intervention before G3 is highly effective
3. Students with **1+ failure + low study time** form the highest risk group
4. **Aspiration for higher education** (`higher = yes`) significantly reduces at-risk probability
5. **Parental education level** (Medu/Fedu) correlates inversely with student risk
6. The model can **flag at-risk students before the final exam**, enabling timely support

---

## 📁 Project Structure

```
student-performance-ml/
│
├── Student_Performance_ML_Project.ipynb   # Main notebook (end-to-end pipeline)
├── student-mat.csv                        # Dataset (UCI)
├── model.pkl                              # Serialized best model
├── app.py                                 # FastAPI deployment script
├── mlruns/                                # MLflow experiment logs
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-username/student-performance-ml.git
cd student-performance-ml

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn fastapi uvicorn joblib mlflow

# Run in Google Colab (recommended)
# Upload student-mat.csv and open the .ipynb notebook
```

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **ML Library** | Scikit-learn |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | FastAPI, Uvicorn |
| **Experiment Tracking** | MLflow |
| **Environment** | Google Colab |

---

## 📈 Results Summary

```
════════════════════════════════════════════════════════════
              📋 FINAL PROJECT SUMMARY
════════════════════════════════════════════════════════════
Dataset    : UCI Student Performance (Math)
Samples    : 395 students | 30+ selected features
Task       : Binary Classification — At-Risk Student Detection
Threshold  : G3 < 10 → At-Risk (1) | G3 ≥ 10 → Pass (0)
Split      : 80% Train / 20% Test (stratified) | 5-Fold CV

🏆 BEST MODEL : Gradient Boosting
   ✅ Accuracy  : ~0.88
   ✅ Precision : ~0.84
   ✅ Recall    : ~0.81
   ✅ F1 Score  : ~0.82
   ✅ ROC-AUC   : ~0.90
════════════════════════════════════════════════════════════
```

---

## 🙋 Author

**Built as part of an Advanced Internship Project — Task 8**  
Production-Level ML Pipeline | End-to-End Data Science

> *"The goal is not just to predict — it's to intervene early enough to make a difference."*

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00E0A4,50:4C3BC4,100:2E2A72&height=120&section=footer"/>

⭐ **Star this repo if you found it useful!** &nbsp;|&nbsp; 🍴 **Fork it to build your own version**

</div>
