import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer

# Set page title
st.title("Prediction of Cardiovascular Risk in New–onset T2D")
st.caption("Based on TyG Index and Carotid Ultrasound Features")

# ===== Load model and data =====
model = joblib.load('LGB.pkl')              # Trained LightGBM model
X_test = pd.read_csv('x_test.csv')          # Original test set for SHAP/LIME context

# ===== Feature list (Displayed names) =====
feature_names = [
    "Age (years)",
    "Hypertension",
    "IMT (mm)",
    "TyG index",
    "Carotid plaque burden",
    "Plaque thickness (mm)"
]

# ===== Input form =====
with st.form("input_form"):
    st.subheader("Please enter the following clinical and ultrasound features:")
    inputs = []

    for col in feature_names:
        if col == "Hypertension":
            inputs.append(st.selectbox(col, options=[0, 1], index=0))

        elif col == "Age (years)":
            min_val = int(X_test["Age"].min())
            max_val = 100
            default_val = int(X_test["Age"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Carotid plaque burden":
            min_val = int(X_test[col].min())
            max_val = 15
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Plaque thickness (mm)":
            min_val = 0.0
            max_val = 7.0
            default_val = float(X_test["Max plaque thickness"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "IMT (mm)":
            min_val = 0.0
            max_val = 1.5
            default_val = float(X_test["IMT"].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "TyG index":
            min_val = 0.0
            max_val = 15.0
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
            )

        else:
            min_val = float(X_test[col].min())
            max_val = float(X_test[col].max())
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val)
            )

    submitted = st.form_submit_button("Submit Prediction")

# ===== Prediction and interpretation =====
if submitted:
    input_data = pd.DataFrame([inputs], columns=feature_names)
    input_data = input_data.round(2)  # 保留两位小数用于显示
    st.subheader("Model Input Features")
    st.dataframe(input_data)

    # Prepare model input with original column names
    model_input = pd.DataFrame([{
        "Age": input_data["Age (years)"].iloc[0],
        "Hypertension": input_data["Hypertension"].iloc[0],
        "IMT": input_data["IMT (mm)"].iloc[0],
        "TyG index": input_data["TyG index"].iloc[0],
        "Carotid plaque burden": input_data["Carotid plaque burden"].iloc[0],
        "Max plaque thickness": input_data["Plaque thickness (mm)"].iloc[0]
    }])

    predicted_proba = model.predict_proba(model_input)[0]
    probability = predicted_proba[1] * 100

    # ==== 整合展示预测结果和 SHAP 可视化 ====
    st.subheader("Prediction Result & Explanation")
    st.markdown(f"**Estimated probability:** {probability:.1f}%")

    # ===== SHAP Force Plot =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):  # Binary classification
        shap_value_sample = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_value_sample = shap_values
        expected_value = explainer.expected_value

    force_plot = shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input,
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()
    st.image("shap_force_plot.png")
