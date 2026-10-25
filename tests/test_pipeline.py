
import os
import numpy as np
import pandas as pd
import pytest

from huggingface_hub import hf_hub_download
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATASET_REPO = "kenshivinay/SuperKart-Sales-Forecasting"
TARGET = "Product_Store_Sales_Total"


@pytest.fixture(scope="module")
def datasets():
    """Download registered datasets once per test session."""

    token = os.getenv("HF_TOKEN")

    train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/train.csv",
        repo_type="dataset",
        token=token
    )

    test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/test.csv",
        repo_type="dataset",
        token=token
    )

    return pd.read_csv(train_path), pd.read_csv(test_path)


def test_datasets_not_empty(datasets):
    """Verify training and testing records are available."""

    train_df, test_df = datasets

    assert not train_df.empty
    assert not test_df.empty


def test_target_column_exists(datasets):
    """Verify the regression target is present."""

    train_df, test_df = datasets

    assert TARGET in train_df.columns
    assert TARGET in test_df.columns


def test_feature_columns_match(datasets):
    """Verify matching input features in train and test."""

    train_df, test_df = datasets

    train_features = train_df.drop(columns=[TARGET])
    test_features = test_df.drop(columns=[TARGET])

    assert list(train_features.columns) == list(test_features.columns)


def test_target_has_no_missing_values(datasets):
    """Verify target values are complete."""

    train_df, test_df = datasets

    assert train_df[TARGET].notna().all()
    assert test_df[TARGET].notna().all()


def test_product_identifier_removed(datasets):
    """Verify unique product identifier was removed."""

    train_df, test_df = datasets

    assert "Product_Id" not in train_df.columns
    assert "Product_Id" not in test_df.columns


def test_preprocessing_and_prediction(datasets):
    """Verify the ML pipeline can preprocess and predict."""

    train_df, _ = datasets

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Use a small sample to keep CI tests fast
    X_sample = X.head(100)
    y_sample = y.head(100)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeRegressor(random_state=42)
        )
    ])

    pipeline.fit(X_sample, y_sample)

    predictions = pipeline.predict(X_sample.head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()

