
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_final_pipeline.joblib"
)

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero p {
        margin-bottom: 0;
        opacity: 0.85;
    }

    .section-label {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }

    .result-panel {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        min-height: 145px;
    }

    .muted {
        opacity: 0.75;
        font-size: 0.9rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Model loading
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>
            An interactive machine-learning demonstration using
            patient health measurements and a trained classification model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational demonstration only. This model is not a medical "
    "diagnostic tool and must not be used to make healthcare decisions."
)


# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About the project")

    st.write(
        "This application uses a Logistic Regression pipeline "
        "trained on the UCI Heart Disease dataset."
    )

    st.divider()

    st.markdown("**Model**")
    st.write("Logistic Regression")

    st.markdown("**Input features**")
    st.write("13")

    st.markdown("**Classification task**")
    st.write("Binary classification")

    st.divider()

    st.caption(
        "The dataset combines records from different sources. "
        "Performance may differ on new patients or hospitals."
    )


# --------------------------------------------------
# 6. Patient form
# --------------------------------------------------

st.header("Patient information")
st.caption("Enter values for all fields, then submit the form.")

with st.form("prediction_form"):
    st.markdown(
        '<div class="section-label">Basic information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=50,
        )

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    with col3:
        cp = st.selectbox(
            "Chest pain type",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic",
            ],
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Clinical measurements</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50,
            max_value=300,
            value=120,
        )

    with col2:
        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=50,
            max_value=700,
            value=200,
        )

    with col3:
        thalch = st.number_input(
            "Maximum heart rate achieved",
            min_value=40,
            max_value=250,
            value=150,
        )

    col1, col2 = st.columns(2)

    with col1:
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl",
            ["False", "True"],
        )

    with col2:
        exang = st.selectbox(
            "Exercise-induced angina",
            ["False", "True"],
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Additional test results</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        restecg = st.selectbox(
            "Resting ECG",
            ["normal", "st-t abnormality", "lv hypertrophy"],
        )

    with col2:
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    with col3:
        slope = st.selectbox(
            "ST segment slope",
            ["upsloping", "flat", "downsloping"],
        )

    col1, col2 = st.columns(2)

    with col1:
        ca = st.selectbox(
            "Number of major vessels (ca)",
            [0, 1, 2, 3],
        )

    with col2:
        thal = st.selectbox(
            "Thalassemia test result",
            ["normal", "fixed defect", "reversable defect"],
        )

    st.divider()

    submitted = st.form_submit_button(
        "Generate prediction",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 7. Prediction and results
# --------------------------------------------------

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs == "True",
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang == "True",
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    try:
        model = load_model()

        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        st.divider()
        st.header("Model output")
        st.caption(
            "The following is a model classification, not a medical assessment."
        )

        result_col, score_col = st.columns([1.5, 1])

        
        with result_col:
             with st.container(border=True):
                if prediction == 1:
                  st.subheader("Class 1")
                  st.write("Model prediction: positive class.")
                else:
                 st.subheader("Class 0")
             st.write("Model prediction: negative class.")

        st.caption(
            "The classification uses the model's default decision threshold."
        )

        with score_col:
            st.metric(
                "Estimated probability for class 1",
                f"{probability:.1%}",
            )
            st.progress(probability)

        st.info(
            "This probability is the model's output for class 1. "
            "It is not a clinically calibrated estimate of an individual's "
            "chance of having heart disease."
        )

        st.warning(
            "This result cannot confirm or rule out heart disease. "
            "Seek assessment from a qualified healthcare professional "
            "for medical concerns."
        )

        with st.expander("View submitted input values"):
            st.dataframe(input_data, use_container_width=True)

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception:
        st.error(
            "Prediction failed. Check the terminal for the error details "
            "and confirm that the saved model matches the input features."
        )
        st.exception(Exception("Prediction inference failed."))
