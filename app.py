import os
import streamlit as st
import pandas as pd

from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.knn import run_knn
from models.naive_bayes import run_naive_bayes
from models.random_forest import run_random_forest
from models.xgboost_model import run_xgboost

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    layout="wide"
)

# -------------------------------------------------
# DARK BLUE THEME + UI FIXES
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- DARK BLUE BACKGROUND ---------- */
    .stApp {
        background-color: #0b1c2d;
        color: #ffffff;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #ffffff;
    }

    /* ---------- PANELS ---------- */
    .panel-box {
        background-color: #12263a;
        border: 1px solid #1f3b5b;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }

    /* ---------- DATAFRAME ---------- */
    .stDataFrame {
        background-color: #12263a;
    }

    /* ---------- UPLOAD BOX TEXT BLACK ---------- */
    div[data-testid="stFileUploader"] * {
        color: #000000 !important;
    }

    /* =================================================
       RUN EVALUATION BUTTON (SECONDARY – FULL CONTROL)
       ================================================= */
    button[kind="secondary"] {
        background-color: #12263a !important;
        border: 2px solid #4f8bf9 !important;
        color: #ffffff !important;
        box-shadow: none !important;
        transition: none !important;
        border-radius: 999px !important;
        padding: 0.6rem 1.4rem !important;
    }

    button[kind="secondary"] span {
        color: #ffffff !important;
        font-weight: 600;
    }

    button[kind="secondary"]:hover,
    button[kind="secondary"]:active,
    button[kind="secondary"]:focus {
        background-color: #12263a !important;
        color: #ffffff !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* ---------- REMOVE DEPLOY + MENU ---------- */
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- REMOVE TOP SPACE ---------- */
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        visibility: hidden;
    }

    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# MAIN HEADER
# -------------------------------------------------
st.markdown(
    """
    <div class="panel-box" style="border-left:6px solid #4f8bf9;">
        <h2>📊 Bank Marketing Classification Dashboard</h2>
        <p>
        Upload test data, select a machine learning model, and
        evaluate performance using standard classification metrics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left_col, right_col = st.columns([1.2, 3.8], gap="large")

# -------------------------------------------------
# LEFT PANEL (MODEL SELECTION)
# -------------------------------------------------
with left_col:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.subheader("⚙ Model Selection")

    model_map = {
        "Logistic Regression": run_logistic_regression,
        "Decision Tree": run_decision_tree,
        "K-Nearest Neighbors": run_knn,
        "Naive Bayes": run_naive_bayes,
        "Random Forest": run_random_forest,
        "XGBoost": run_xgboost
    }

    selected_model = st.selectbox(
        "Choose a Model",
        list(model_map.keys())
    )

    st.markdown(
        "<small>Select one classification algorithm.   Click Run Evaluation Below </small>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# RIGHT PANEL (UPLOAD + RESULTS)
# -------------------------------------------------
with right_col:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)

    st.markdown(
        "<h4 style='color:black;'>📁 Upload Test Dataset</h4>",
        unsafe_allow_html=True
    )

    # ---------- DOWNLOAD SAMPLE CSV BUTTON ----------
    sample_csv_path = "bank.csv"

    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Sample Dataset (bank.csv)",
                data=f,
                file_name="bank.csv",
                mime="text/csv"
            )
    else:
        st.warning("Sample dataset (bank.csv) not found in root directory.")

    # ---------- FILE UPLOADER ----------
    uploaded_file = st.file_uploader(
        "CSV file (test data only)",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";")

            st.subheader("📄 Dataset Preview")
            # 👇 Show only first 10 rows
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 Run Evaluation", type="secondary"):
                with st.spinner("Evaluating model..."):
                    metrics_df, report_df, _ = model_map[selected_model](df)

                st.success("Evaluation completed successfully!")

                st.subheader("📈 Evaluation Metrics")
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("📋 Classification Report")
                st.dataframe(report_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file to begin.")

    st.markdown('</div>', unsafe_allow_html=True)