# Personalized Cardiovascular Risk Assessment

## 📖 Project Overview

This repository hosts a machine learning framework for personalized cardiovascular risk assessment using clinical tabular health metrics and explainable artificial intelligence. The deployed system leverages an ensemble **Random Forest Classifier** and **Explainable AI (TreeSHAP)** to deliver accurate, personalized, and interpretable cardiac risk predictions.

During the baseline benchmarking phase, three algorithms were rigorously evaluated: **Logistic Regression**, **Random Forest**, and **XGBoost**. The **Random Forest** model was selected as champion based on superior clinical **Recall (90.48% on the hold-out test set)** and **ROC-AUC (0.9362)**, ensuring minimal false-negative risk. The complete inference pipeline is deployed via an interactive **Streamlit** application.

---

## 🔗 Dataset Sources & External Links

### 1. UCI Heart Disease Dataset
* **Source**: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
* **Mirror / Alternative**: [Kaggle UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
* **Description**: Contains clinical patient records from the Cleveland database with 13 key diagnostic features and binary cardiovascular status.

### 2. Diagnostic ECG Repository (Raw Multi-Modal Data)
* **Primary Source (Mendeley Data)**: [ECG Images Dataset of Cardiac Patients (DOI: 10.17632/gwbz3fsgp8.2)](https://data.mendeley.com/datasets/gwbz3fsgp8/2)
* **Description**: Collected for multi-modal exploratory research across diagnostic classes (Myocardial Infarction, Abnormal Heartbeat, Normal controls). *Note: The active deployed risk prediction pipeline utilizes the 13 verified clinical tabular metrics.*

---

## 📁 Raw Dataset Structure (`data/raw/`)

All raw datasets and diagnostic images are organized within the `data/raw` directory:

```text
data/
└── raw/
    ├── UCI_Heart_Disease.csv # Canonical UCI Cleveland tabular dataset
    ├── ECG/                  # Myocardial Infarction (MI) 12-lead ECG images
    ├── abnormal_heatbeat/    # Abnormal Heartbeat ECG images
    ├── his_MI/               # Previous History of Myocardial Infarction (PMI) ECG images
    ├── normal/               # Healthy / Normal control ECG images
    ├── uci - normal/         # UCI Heart Disease - Normal patient profile samples
    └── uci - sick/           # UCI Heart Disease - Diagnosed disease patient profile samples
```

---

## 📊 Tabular Clinical Features Specification

The 13 clinical attributes used for model training and real-time inference:

| # | Feature | Description | Values / Units |
|---|---------|-------------|----------------|
| 1 | `age` | Patient age | Continuous (years) |
| 2 | `sex` | Biological sex | `1` = Male, `0` = Female |
| 3 | `cp` | Chest pain type | `1`: Typical angina, `2`: Atypical angina, `3`: Non-anginal pain, `4`: Asymptomatic |
| 4 | `trestbps` | Resting blood pressure | Continuous (mm Hg on admission) |
| 5 | `chol` | Serum cholesterol | Continuous (mg/dl) |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | `1` = True, `0` = False |
| 7 | `restecg` | Resting electrocardiographic results | `0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | Continuous (bpm) |
| 9 | `exang` | Exercise-induced angina | `1` = Yes, `0` = No |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Continuous (0.0 to 6.2) |
| 11 | `slope` | Slope of peak exercise ST segment | `1`: Upsloping, `2`: Flat, `3`: Downsloping |
| 12 | `ca` | Major vessels colored by fluoroscopy | `0` to `3` |
| 13 | `thal` | Thalassemia status | `3` = Normal, `6` = Fixed defect, `7` = Reversible defect |

**Target Variable**: `target_binary` (`0`: No heart disease, `1`: Heart disease present).

---

## 🚀 Machine Learning & Inference Pipeline

```text
Dataset Collection (UCI Cleveland Heart Disease Dataset)
      │
      ▼
Data Preprocessing, Deduplication & Mode Imputation (ca: 0.0, thal: 3.0)
      │
      ▼
Stratified Train / Validation / Test Splitting (70% / 15% / 15%)
      │
      ▼
Baseline Model Benchmarking (Logistic Regression, Random Forest, XGBoost)
      │
      ▼
Champion Selection: Random Forest (Top Validation Recall & ROC-AUC)
      │
      ▼
Hold-Out Test Evaluation (Accuracy: 86.96%, Recall: 90.48%, ROC-AUC: 0.9362)
      │
      ▼
Model Explainability (TreeSHAP Individual Patient Feature Attribution)
      │
      ▼
Interactive Streamlit Application (Presentation Bands: LOW / MEDIUM / HIGH)
```

---

## 📈 Model Performance & Evaluation Metrics

Evaluated on the stratified partitions (`random_state=42`):

### 1. Validation Set Benchmarking (45 Patients)
| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (Champion)** | **84.44%** | **85.00%** | **80.95%** | **82.93%** | **0.9256** |
| Logistic Regression | 80.00% | 83.33% | 71.43% | 76.92% | 0.8889 |
| XGBoost | 77.78% | 78.95% | 71.43% | 75.00% | 0.8948 |

### 2. Final Hold-Out Test Set Performance (46 Patients, Random Forest)
* **Accuracy**: **86.96%** (40 / 46 correct classifications)
* **Recall (Sensitivity)**: **90.48%** (19 of 21 cardiac disease cases detected; **only 2 false negatives**)
* **Precision**: **82.61%**
* **F1-Score**: **86.36%**
* **ROC-AUC**: **0.9362**

---

## 🔍 Explainability (TreeSHAP)

For every individual patient prediction, local feature attributions are computed via TreeSHAP:
$$f(x) = \mathbb{E}[f(x)] + \sum_{j=1}^{13} \phi_j$$
* **Positive Contribution ($\phi_j > 0$)**: Feature pushed model prediction toward higher disease probability (`increases risk`).
* **Negative Contribution ($\phi_j < 0$)**: Feature pushed model prediction toward lower disease probability (`decreases risk`).
* *Note: SHAP values describe statistical model contributions and do not represent biological or clinical causation.*

---

## 💻 Running the Streamlit Application

```bash
streamlit run app.py
```
Access the application locally at `http://localhost:8501`.

---

## ⚠️ Important Educational & Research Disclaimer

This application and codebase are developed for **educational, academic, and research demonstration purposes only**. The risk scores are generated by a machine learning model and do **not** constitute a clinical diagnosis, medical evaluation, treatment plan, or medication prescription. The **LOW / MEDIUM / HIGH** risk categories are project-defined presentation-layer bands and are **not** clinically validated diagnostic thresholds. Always consult a qualified medical professional for health evaluations.