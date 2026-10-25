
import os
import json
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download

# Application configuration
st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)

MODEL_REPO = os.getenv(
    "MODEL_REPO_ID",
    "kenshivinay/SuperKart-Sales-Model"
)

# Read input feature configuration
with open(os.path.join(os.path.dirname(__file__), "input_schema.json"), "r", encoding="utf-8") as f:
    schema = json.load(f)


@st.cache_resource
def load_model():
    """Download and load the registered ML pipeline."""

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except (KeyError, FileNotFoundError):
            pass

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_superkart_model.joblib",
        repo_type="model",
        token=hf_token
    )

    # Load only our own trusted model artifact
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the registered prediction model.")
    st.exception(error)
    st.stop()


# Application header
st.title("🛒 SuperKart Sales Prediction")

st.markdown(
    """
    Predict product-store sales revenue using machine learning.

    Enter product and store information below and click
    **Predict Sales** to generate an estimate.
    """
)

st.divider()

st.subheader("Product and Store Information")

# Capture user inputs
user_inputs = {}

with st.form("prediction_form"):

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(schema["feature_columns"]):

        container = (
            left_column if index % 2 == 0
            else right_column
        )

        label = feature.replace("_", " ")

        with container:

            if feature in schema["categorical"]:

                options = schema["categorical"][feature]

                user_inputs[feature] = st.selectbox(
                    label,
                    options=options
                )

            else:

                config = schema["numerical"][feature]

                if config["integer"]:

                    user_inputs[feature] = st.number_input(
                        label,
                        value=int(config["default"]),
                        step=1
                    )

                else:

                    user_inputs[feature] = st.number_input(
                        label,
                        value=float(config["default"]),
                        step=0.01,
                        format="%.4f"
                    )

    submitted = st.form_submit_button(
        "Predict Sales",
        type="primary",
        use_container_width=True
    )


# Generate prediction
if submitted:

    # Convert entered values to DataFrame
    input_df = pd.DataFrame(
        [user_inputs]
    )

    # Ensure correct feature order
    input_df = input_df[
        schema["feature_columns"]
    ]

    try:

        # Generate sales prediction
        prediction = model.predict(input_df)[0]

        # Display predicted sales
        st.success("Prediction generated successfully!")

        st.metric(
            label="Predicted Sales Revenue",
            value=f"{prediction:,.2f}"
        )

        # Show input record
        with st.expander("View Prediction Input Data"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:

        st.error("Prediction failed.")
        st.exception(error)


st.divider()

st.caption(
    "SuperKart MLOps Project | "
    "Machine Learning Sales Prediction"
)
