
"""
SuperKart MLOps - Automated Data Preparation

Steps:
1. Download original dataset from Hugging Face.
2. Perform data cleaning.
3. Remove unnecessary columns.
4. Split into training and testing datasets.
5. Save datasets locally.
6. Upload processed datasets to Hugging Face.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Project Configuration
# --------------------------------------------------

DATASET_REPO = "kenshivinay/SuperKart-Sales-Forecasting"

TARGET = "Product_Store_Sales_Total"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# GitHub Actions will provide this token as a secret
HF_TOKEN = os.getenv("HF_TOKEN")

# Save files inside the project's data folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Data Preparation Function
# --------------------------------------------------

def prepare_data():

    print("Starting SuperKart data preparation...")

    # ----------------------------------------------
    # Step 1: Download dataset from Hugging Face
    # ----------------------------------------------

    dataset_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="SuperKart.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )

    df = pd.read_csv(dataset_path)

    print("Dataset downloaded successfully!")
    print("Original dataset shape:", df.shape)

    # ----------------------------------------------
    # Step 2: Data Cleaning
    # ----------------------------------------------

    df_clean = df.copy()

    # Remove unnecessary product identifier
    df_clean.drop(
        columns=["Product_Id"],
        inplace=True,
        errors="ignore"
    )

    # Standardize sugar content categories
    df_clean["Product_Sugar_Content"] = (
        df_clean["Product_Sugar_Content"]
        .str.strip()
        .replace({"reg": "Regular"})
    )

    # Remove exact duplicate records
    df_clean.drop_duplicates(inplace=True)

    # Remove rows with missing target values
# Ensure target contains valid numerical values
df_clean[TARGET] = pd.to_numeric(
    df_clean[TARGET],
    errors="coerce"
)

# Replace infinite target values with NaN
df_clean[TARGET] = df_clean[TARGET].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove records with invalid target values
df_clean.dropna(
    subset=[TARGET],
    inplace=True
)
    # Convert numerical columns to correct data types
    numeric_columns = [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Establishment_Year"
    ]

    for column in numeric_columns:
        df_clean[column] = pd.to_numeric(
            df_clean[column],
            errors="coerce"
        )

    # Handle infinite numerical values
    df_clean.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Verify cleaned dataset
    print("\nCleaned dataset shape:", df_clean.shape)

    if df_clean.empty:
        raise ValueError("Cleaned dataset is empty!")

    # ----------------------------------------------
    # Step 3: Train-Test Split
    # ----------------------------------------------

    train_df, test_df = train_test_split(
        df_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Reset indexes
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    # ----------------------------------------------
    # Step 4: Save Datasets Locally
    # ----------------------------------------------

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nTraining dataset saved:", train_path)
    print("Testing dataset saved:", test_path)

    # ----------------------------------------------
    # Step 5: Upload Datasets to Hugging Face
    # ----------------------------------------------

    # Upload requires a write-enabled token
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Configure it in GitHub Actions secrets."
        )

    api = HfApi(token=HF_TOKEN)

    api.upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="data/train.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Automated training dataset update"
    )

    api.upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo="data/test.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Automated testing dataset update"
    )

    print("\nDatasets uploaded to Hugging Face!")
    print("Data preparation completed successfully!")


# --------------------------------------------------
# 3. Execute Script
# --------------------------------------------------

if __name__ == "__main__":
    prepare_data()
