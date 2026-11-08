
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Classification")
st.write(
    "Compare five Machine Learning classification models "
    "using the same test dataset."
)

# --------------------------------------------------
# 2. MODEL FILES AND FEATURES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# --------------------------------------------------
# 3. LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(filename):
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)

# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

# --------------------------------------------------
# 5. CSV UPLOAD
# --------------------------------------------------

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    key="test_csv_upload"
)

if uploaded_file is not None:
    try:
        st.session_state.uploaded_data = pd.read_csv(
            uploaded_file
        )
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

# --------------------------------------------------
# 6. MODEL SELECTION
# --------------------------------------------------

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys()),
    key="selected_model"
)

st.sidebar.info(
    "Upload the dataset once, then switch models "
    "to update the results automatically."
)

# --------------------------------------------------
# 7. PROCESS DATA
# --------------------------------------------------

if st.session_state.uploaded_data is not None:

    df = st.session_state.uploaded_data.copy()

    st.subheader("Uploaded Dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("Total Rows:", df.shape[0])
    st.write("Total Columns:", df.shape[1])

    # Validate required features
    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing feature columns: {missing_columns}"
        )
        st.stop()

    # Prepare input features
    X = df[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # 8. LOAD SELECTED MODEL
    # --------------------------------------------------

    try:
        model = load_model(
            MODEL_FILES[model_name]
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.success(
        f"Currently displaying results for: {model_name}"
    )

    # --------------------------------------------------
    # 9. GENERATE PREDICTIONS
    # --------------------------------------------------

    try:
        predictions = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    output = df.copy()
    output["prediction"] = predictions

    st.subheader("Prediction Results")

    st.dataframe(
        output,
        use_container_width=True
    )

    st.download_button(
        label="Download Predictions CSV",
        data=output.to_csv(index=False),
        file_name=(
            model_name.lower().replace(" ", "_")
            + "_predictions.csv"
        ),
        mime="text/csv"
    )

    # --------------------------------------------------
    # 10. EVALUATION METRICS
    # --------------------------------------------------

    if "target" in df.columns:

        y_true = pd.to_numeric(
            df["target"],
            errors="coerce"
        )

        valid_labels = (
            y_true.notna()
            & y_true.isin([0, 1])
        )

        if not valid_labels.all():
            st.warning(
                "The target column must contain "
                "only 0 and 1, with no missing values."
            )
            st.stop()

        y_true = y_true.astype(int)

        try:
            probabilities = model.predict_proba(X)[:, 1]
        except Exception as e:
            st.error(
                f"Could not calculate probabilities: {e}"
            )
            st.stop()

        accuracy = accuracy_score(
            y_true, predictions
        )

        precision = precision_score(
            y_true, predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true, predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true, predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true, predictions
        )

        if y_true.nunique() == 2:
            auc = roc_auc_score(
                y_true, probabilities
            )
        else:
            auc = None

        # --------------------------------------------------
        # 11. DISPLAY SIX METRICS
        # --------------------------------------------------

        st.subheader(
            f"Evaluation Metrics — {model_name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}" if auc is not None else "N/A"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # --------------------------------------------------
        # 12. CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader(
            f"Confusion Matrix — {model_name}"
        )

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(
            f"{model_name} Confusion Matrix"
        )

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # 13. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader(
            f"Classification Report — {model_name}"
        )

        report = classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        # --------------------------------------------------
        # 14. SUMMARY
        # --------------------------------------------------

        st.subheader("Model Summary")

        st.write("Selected Model:", model_name)
        st.write("Total Test Samples:", len(X))
        st.write("Correct Predictions:", int(
            (predictions == y_true).sum()
        ))
        st.write("Incorrect Predictions:", int(
            (predictions != y_true).sum()
        ))

    else:
        st.info(
            "The uploaded CSV has no target column. "
            "Predictions are available, but evaluation "
            "metrics require actual target labels."
        )

else:
    st.info(
        "Please upload test_data.csv using the sidebar "
        "to start evaluating classification models."
    )
