"""
Personalized Cardiovascular Risk Assessment Application
Streamlit Interactive Clinical Decision Support & Explainable AI Interface
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd

from src.predict import (
    predict_cardiovascular_risk,
    get_patient_summary,
    FEATURE_NAMES
)
from src.explain import (
    explain_prediction,
    plot_patient_shap
)

# Page configuration
st.set_page_config(
    page_title="Personalized Cardiovascular Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling for a clean, modern, student/faculty presentation theme
st.markdown("""
<style>
    .main-header {
        text-align: left;
        padding: 0.5rem 0 1rem 0;
    }
    .main-title {
        color: #d90429;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4a5568;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    .header-desc {
        color: #718096;
        font-size: 0.98rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-card-low {
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
        border: 2px solid #38a169;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card-medium {
        background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
        border: 2px solid #dd6b20;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card-high {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #e53e3e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-heading {
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    .result-prob {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .result-badge {
        display: inline-block;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.35rem 1.25rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .badge-low { background-color: #38a169; color: white; }
    .badge-medium { background-color: #dd6b20; color: white; }
    .badge-high { background-color: #e53e3e; color: white; }
    .result-status {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 0.5rem;
    }
    .statement-box {
        padding: 0.85rem 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 0.75rem 0;
    }
    .statement-low { background-color: #edf2f7; color: #2d3748; border-left: 4px solid #38a169; }
    .statement-high { background-color: #fff5f5; color: #9b2c2c; border-left: 4px solid #e53e3e; }
    .factor-item {
        padding: 0.65rem 0.85rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .factor-increase {
        background-color: #fff5f5;
        border-left: 3px solid #e53e3e;
    }
    .factor-decrease {
        background-color: #f0fff4;
        border-left: 3px solid #38a169;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    st.markdown("""
    <div class="main-header">
        <div class="main-title">❤️ Personalized Cardiovascular Risk Assessment</div>
        <div class="sub-title">Machine Learning based cardiovascular risk prediction with explainable AI</div>
        <div class="header-desc">
            Enter the patient's cardiovascular parameters below to estimate the model-predicted probability
            of heart disease and understand the factors influencing the prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Patient Input Section
    # ---------------------------------------------------------
    st.subheader("📋 Patient Information")

    with st.container():
        # --- Basic Information ---
        st.markdown("##### 👤 Basic Information")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            age = st.number_input(
                "Age (years)",
                min_value=20,
                max_value=100,
                value=54,
                step=1,
                help="Patient's age in years (range: 20–100)"
            )
        with b_col2:
            sex_choice = st.selectbox(
                "Biological Sex",
                options=[("Male", 1.0), ("Female", 0.0)],
                format_func=lambda x: x[0],
                help="Biological sex of the patient"
            )
            sex = sex_choice[1]

        # --- Clinical Measurements ---
        st.markdown("##### 🩺 Clinical Measurements")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            trestbps = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=70,
                max_value=250,
                value=130,
                step=1,
                help="Resting blood pressure in mm Hg upon hospital admission"
            )
        with c_col2:
            chol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=230,
                step=1,
                help="Serum cholesterol in mg/dL"
            )
        with c_col3:
            thalach = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=60,
                max_value=230,
                value=150,
                step=1,
                help="Maximum heart rate achieved during exercise stress test"
            )
        with c_col4:
            oldpeak = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.2,
                step=0.1,
                format="%.1f",
                help="ST depression induced by exercise relative to rest"
            )

        # --- Medical / Test Information ---
        st.markdown("##### 🔬 Diagnostic Test Results")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            cp_choice = st.selectbox(
                "Chest Pain Type",
                options=[
                    ("1 = Typical Angina", 1.0),
                    ("2 = Atypical Angina", 2.0),
                    ("3 = Non-anginal Pain", 3.0),
                    ("4 = Asymptomatic", 4.0)
                ],
                index=2,
                format_func=lambda x: x[0],
                help="Clinical classification of chest pain"
            )
            cp = cp_choice[1]
        with m_col2:
            restecg_choice = st.selectbox(
                "Resting ECG",
                options=[
                    ("0 = Normal", 0.0),
                    ("1 = ST-T Wave Abnormality", 1.0),
                    ("2 = Left Ventricular Hypertrophy", 2.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Resting electrocardiographic results"
            )
            restecg = restecg_choice[1]
        with m_col3:
            fbs_choice = st.selectbox(
                "Fasting Blood Sugar",
                options=[
                    ("No (≤ 120 mg/dL)", 0.0),
                    ("Yes (> 120 mg/dL)", 1.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Fasting blood sugar > 120 mg/dL"
            )
            fbs = fbs_choice[1]
        with m_col4:
            exang_choice = st.selectbox(
                "Exercise-Induced Angina",
                options=[
                    ("No", 0.0),
                    ("Yes", 1.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Angina provoked by exercise test"
            )
            exang = exang_choice[1]

        # --- Additional Findings ---
        st.markdown("##### 🔍 Additional Angiographic & Stress Findings")
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            slope_choice = st.selectbox(
                "ST Segment Slope",
                options=[
                    ("1 = Upsloping", 1.0),
                    ("2 = Flat", 2.0),
                    ("3 = Downsloping", 3.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Slope of the peak exercise ST segment"
            )
            slope = slope_choice[1]
        with a_col2:
            ca_choice = st.selectbox(
                "Major Vessels (Fluoroscopy)",
                options=[
                    ("0 vessels", 0.0),
                    ("1 vessel", 1.0),
                    ("2 vessels", 2.0),
                    ("3 vessels", 3.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Number of major vessels (0–3) colored by fluoroscopy"
            )
            ca = ca_choice[1]
        with a_col3:
            thal_choice = st.selectbox(
                "Thalassemia Status",
                options=[
                    ("3 = Normal", 3.0),
                    ("6 = Fixed Defect", 6.0),
                    ("7 = Reversible Defect", 7.0)
                ],
                index=0,
                format_func=lambda x: x[0],
                help="Thallium stress scintigraphy result (verified project encoding: 3, 6, 7)"
            )
            thal = thal_choice[1]

    # ---------------------------------------------------------
    # Predict Button
    # ---------------------------------------------------------
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Assess Cardiovascular Risk", type="primary", use_container_width=True)

    # Input Validation & State Update
    if predict_clicked:
        try:
            # Build patient dictionary strictly with the 13 required features
            patient_data = {
                "age": float(age),
                "sex": float(sex),
                "cp": float(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": float(fbs),
                "restecg": float(restecg),
                "thalach": float(thalach),
                "exang": float(exang),
                "oldpeak": float(oldpeak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": float(thal)
            }

            # Run inference
            pred_result = predict_cardiovascular_risk(patient_data)
            patient_summary = get_patient_summary(patient_data)
            shap_explanations = explain_prediction(patient_data, top_n=5)

            # Generate and save SHAP plot
            shap_plot_path = "outputs/shap_patient_latest.png"
            plot_patient_shap(patient_data, output_path=shap_plot_path, top_n=5)

            # Save in session state
            st.session_state["assessment_result"] = {
                "patient_data": patient_data,
                "pred_result": pred_result,
                "patient_summary": patient_summary,
                "shap_explanations": shap_explanations,
                "shap_plot_path": shap_plot_path
            }

        except Exception as e:
            st.error(f"Error executing cardiovascular risk assessment: {e}")

    # ---------------------------------------------------------
    # Results Presentation Section
    # ---------------------------------------------------------
    if "assessment_result" in st.session_state:
        res = st.session_state["assessment_result"]
        pred_res = res["pred_result"]
        summary = res["patient_summary"]
        shap_factors = res["shap_explanations"]
        shap_img = res["shap_plot_path"]

        st.markdown("---")
        st.subheader("📊 Assessment Results")

        # Result Card Layout
        prob_pct = pred_res["disease_probability_percent"]
        risk_level = pred_res["risk_level"]
        status_label = pred_res["prediction_label"]
        pred_class = pred_res["predicted_class"]

        card_class = f"result-card-{risk_level.lower()}"
        badge_class = f"badge-{risk_level.lower()}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="result-heading">Cardiovascular Risk Assessment Result</div>
            <div class="result-prob">{prob_pct:.1f}%</div>
            <div class="result-badge {badge_class}">Risk Level: {risk_level}</div>
            <div class="result-status">Predicted Status: {status_label}</div>
            <div style="font-size: 0.95rem; color: #4a5568; margin-top: 0.5rem;">
                <strong>Predicted Probability:</strong> {prob_pct:.1f}%
            </div>
            <div style="font-size: 0.8rem; color: #718096; margin-top: 0.5rem; font-style: italic;">
                Risk level is a project-defined presentation band based on model-predicted probability and is not a clinical diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic Prediction Statement
        if pred_class == 1:
            st.markdown("""
            <div class="statement-box statement-high">
                ⚠️ <strong>Model Assessment:</strong> The model predicts a <strong>higher likelihood</strong> of cardiovascular disease for the provided patient profile.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="statement-box statement-low">
                ℹ️ <strong>Model Assessment:</strong> The model predicts a <strong>lower likelihood</strong> of cardiovascular disease for the provided patient profile.
            </div>
            """, unsafe_allow_html=True)

        # Two-column layout: Patient Summary & Key Contributing Factors
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown("#### 👤 Patient Summary")
            summary_df = pd.DataFrame([
                {"Parameter": k, "Entered Value": v} for k, v in summary.items()
            ])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("#### 🔍 Key Contributing Factors (SHAP)")
            st.caption("Top 5 factors influencing this specific model prediction:")

            for factor in shap_factors:
                d_name = factor["display_name"]
                s_val = factor["shap_value"]
                direction = factor["direction"]

                if direction == "increases risk":
                    icon = "🟠"
                    desc = "Contributed toward higher model-predicted risk"
                    f_class = "factor-increase"
                    sign_str = f"+{s_val:.4f}"
                else:
                    icon = "🟢"
                    desc = "Contributed toward lower model-predicted risk"
                    f_class = "factor-decrease"
                    sign_str = f"{s_val:.4f}"

                st.markdown(f"""
                <div class="factor-item {f_class}">
                    <strong>{icon} {d_name}</strong> ({sign_str})<br>
                    <span style="font-size: 0.86rem; color: #4a5568;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        # SHAP Chart Display
        st.markdown("#### 📈 Individual Feature Attribution Chart")
        if os.path.exists(shap_img):
            st.image(shap_img, use_container_width=True)

        # General Recommendations
        st.markdown("---")
        st.markdown("#### 💡 General Recommendations")

        if risk_level == "LOW":
            st.info("""
            **General Wellness Guidance for Low Risk Profile:**
            * Maintain regular, moderate physical activity (e.g., 150 minutes of aerobic exercise weekly).
            * Follow a balanced, heart-healthy dietary pattern (rich in whole grains, vegetables, and lean proteins).
            * Monitor blood pressure and cholesterol periodically during routine check-ups.
            * Avoid smoking and minimize exposure to secondhand smoke.
            """)
        elif risk_level == "MEDIUM":
            st.warning("""
            **General Guidance for Medium Risk Profile:**
            * Monitor blood pressure and serum cholesterol regularly.
            * Maintain regular physical activity as appropriate for your fitness level.
            * Prefer a balanced diet lower in excess sodium and saturated/trans fats.
            * Avoid smoking and moderate or eliminate alcohol consumption.
            * Consider discussing cardiovascular risk factors with a qualified healthcare professional.
            """)
        else:  # HIGH
            st.error("""
            **General Guidance for Elevated Risk Profile:**
            * Consider seeking a comprehensive medical evaluation from a qualified physician or cardiologist.
            * Actively monitor blood pressure, resting heart rate, and lipid panels.
            * Maintain heart-healthy lifestyle modifications under clinical supervision.
            * Avoid smoking and strenuous unaccustomed physical exertion prior to clinical clearance.
            * Strictly adhere to guidance and treatment plans provided by licensed healthcare professionals.
            """)

    # ---------------------------------------------------------
    # Important Disclaimer Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Disclaimer**

    This application is developed for **educational and research purposes only**. The prediction is generated by a machine learning model and should **not** be considered a medical diagnosis, clinical decision, or substitute for professional medical advice.

    The **LOW / MEDIUM / HIGH** risk bands are project-defined presentation categories and are **not** clinically validated risk thresholds. Always consult a qualified medical provider for diagnostic evaluation.
    """)

    # ---------------------------------------------------------
    # About the Model Expander
    # ---------------------------------------------------------
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        * **Algorithm**: Random Forest Classifier (100 Decision Trees, unscaled feature splits)
        * **Dataset**: UCI Cleveland Heart Disease Dataset (303 records, deduplicated, clean mode imputation)
        * **Input Features**: 13 clinical & diagnostic metrics (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`)
        * **Target**: Binary cardiovascular disease prediction (`0`: healthy / <50% stenosis, `1`: cardiac disease / >50% stenosis)
        * **Validation Accuracy**: **84.44%**
        * **Validation Recall**: **80.95%**
        * **Validation ROC-AUC**: **0.9256**
        * **Test Accuracy (Hold-out)**: **86.96%**
        * **Test Recall / Sensitivity (Hold-out)**: **90.48%** (19 / 21 disease cases detected)
        * **Test ROC-AUC**: **0.9362**
        * **Explainability Engine**: TreeSHAP (SHapley Additive exPlanations)

        *Note: The metrics above reflect evaluation results on this project's academic dataset partition and do not imply regulatory or clinical validation.*
        """)


if __name__ == "__main__":
    main()
