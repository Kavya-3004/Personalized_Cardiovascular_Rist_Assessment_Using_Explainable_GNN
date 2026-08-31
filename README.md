# Personalized Cardiovascular Risk Assessment Using Explainable Graph Neural Networks

## 📖 Project Overview

This repository hosts a multi-modal cardiovascular risk assessment framework that combines clinical tabular health metrics with diagnostic electrocardiogram (ECG) imagery. The system leverages **Graph Neural Networks (GNNs)**, **XGBoost**, and **Explainable AI (SHAP / GNNExplainer)** to deliver accurate, personalized, and interpretable cardiac risk predictions.

---

## 🔗 Dataset Sources & External Links

### 1. UCI Heart Disease Dataset
* **Source**: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
* **Mirror / Alternative**: [Kaggle UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
* **Description**: Contains clinical patient records from four databases (Cleveland, Hungary, Switzerland, and Long Beach V.A.) with 14 key diagnostic features.

### 2. ECG Images Dataset of Cardiac Patients
* **Primary Source (Mendeley Data)**: [ECG Images Dataset of Cardiac Patients (DOI: 10.17632/gwbz3fsgp8.2)](https://data.mendeley.com/datasets/gwbz3fsgp8/2)
* **Kaggle Mirror**: [Kaggle ECG Image Dataset](https://www.kaggle.com/datasets/evilvirus7/ecg-image-dataset)
* **Alternative ECG Repository**: [Kaggle ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
* **Description**: High-resolution 12-lead ECG image recordings categorized into diagnostic classes including Myocardial Infarction, Previous History of MI, Abnormal Heartbeat, and Normal controls.

### 3. Comprehensive Heart Disease Indicators (Kaggle)
* **Source**: [Kaggle Heart Disease Health Indicators](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators)
* **Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith82/heart-disease-dataset)

---

## 📁 Raw Dataset Structure (`data/raw/`)

All raw datasets and diagnostic images are organized within the [`data/raw`](../data/raw) directory:

```text
data/
└── raw/
    ├── ECG/                  # Myocardial Infarction (MI) 12-lead ECG images
    ├── abnormal_heatbeat/    # Abnormal Heartbeat ECG images
    ├── his_MI/               # Previous History of Myocardial Infarction (PMI) ECG images
    ├── normal/               # Healthy / Normal control ECG images
    ├── uci - normal/         # UCI Heart Disease - Normal patient profile samples
    └── uci - sick/           # UCI Heart Disease - Diagnosed disease patient profile samples
```

### Direct Links to Local Subdirectories:
* [📂 `data/raw/ECG`](../data/raw/ECG): Contains ECG images labeled for acute Myocardial Infarction (`MI(1).jpg`, `MI(2).jpg`, ...).
* [📂 `data/raw/abnormal_heatbeat`](../data/raw/abnormal_heatbeat): Contains ECG images demonstrating arrhythmias and abnormal heartbeats (`HB(1).jpg`, `HB(2).jpg`, ...).
* [📂 `data/raw/his_MI`](../data/raw/his_MI): Contains ECG images demonstrating previous history of MI (`PMI(1).jpg`, `PMI(2).jpg`, ...).
* [📂 `data/raw/normal`](../data/raw/normal): Contains baseline normal healthy ECG recordings (`Normal(1).jpg`, `Normal(2).jpg`, ...).
* [📂 `data/raw/uci - normal`](../data/raw/uci%20-%20normal): Tabular/visual profile representations for non-disease cases (`img0001--5.29443.jpg`, ...).
* [📂 `data/raw/uci - sick`](../data/raw/uci%20-%20sick): Tabular/visual profile representations for positive heart disease cases (`IM00001.jpg`, ...).

---

## 📊 Tabular Clinical Features Specification

The clinical attributes present in the UCI / Cleveland Heart Disease data:

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
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Continuous |
| 11 | `slope` | Slope of peak exercise ST segment | `1`: Upsloping, `2`: Flat, `3`: Downsloping |
| 12 | `ca` | Major vessels colored by fluoroscopy | `0` to `3` |
| 13 | `thal` | Thalassemia | `3` = Normal, `6` = Fixed defect, `7` = Reversible defect |
| 14 | `target` | Diagnosis of heart disease | `0`: No disease (< 50% diameter narrowing), `1-4`: Disease (> 50% narrowing) |

---

## 🚀 Project Pipeline

```text
Dataset Collection (Tabular + ECG Images)
      │
      ▼
Data Preprocessing & Missing Value Imputation
      │
      ▼
Patient Graph Construction (k-NN / Clinical Similarity)
      │
      ▼
Graph Neural Network Modeling (GCN / GAT / GraphSAGE)
      │
      ▼
Baseline ML Benchmark (XGBoost, Random Forest)
      │
      ▼
Model Interpretability (SHAP & GNNExplainer)
      │
      ▼
Clinical Evaluation & Risk Assessment
```

---

## 🛠️ GitHub Dataset Upload & Git LFS Guide

When pushing datasets containing images or large `.csv` files to GitHub:

1. **Install Git LFS** (if not already installed):
   ```bash
   git lfs install
   ```

2. **Track image and data formats**:
   ```bash
   git lfs track "*.jpg"
   git lfs track "*.png"
   git lfs track "*.csv"
   ```

3. **Stage and commit**:
   ```bash
   git add .gitattributes
   git add data/
   git commit -m "Add raw datasets and ECG images"
   git push origin main
   ```