
import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_REPO = "kenshivinay/SuperKart-Sales-Forecasting"
MODEL_REPO = "kenshivinay/SuperKart-Sales-Model"

TARGET = "Product_Store_Sales_Total"
RANDOM_STATE = 42

# Number of hyperparameter combinations
N_ITER = int(os.getenv("N_ITER", "10"))

# Hugging Face authentication
HF_TOKEN = os.getenv("HF_TOKEN")

# Local artifact directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    print("Starting automated model training...")

    # ----------------------------------------------
    # 1. Load registered training and testing data
    # ----------------------------------------------

    train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/train.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )

    test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/test.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    # ----------------------------------------------
    # 2. Separate features and target
    # ----------------------------------------------

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    assert list(X_train.columns) == list(X_test.columns)

    # ----------------------------------------------
    # 3. Define feature preprocessing
    # ----------------------------------------------

    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
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

    # ----------------------------------------------
    # 4. Define regression algorithms
    # ----------------------------------------------

    models = {
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist"
        )
    }

    # ----------------------------------------------
    # 5. Define hyperparameter search spaces
    # ----------------------------------------------

    param_grids = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.7, 1.0]
        },

        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0]
        },

        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    }

    # ----------------------------------------------
    # 6. Hyperparameter tuning
    # ----------------------------------------------

    results = []
    trained_models = {}

    for name, estimator in models.items():

        print(f"\nTraining: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=N_ITER,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=2,
            refit=True,
            verbose=1
        )

        search.fit(X_train, y_train)

        trained_models[name] = search.best_estimator_

        results.append({
            "Model": name,
            "CV_RMSE": float(-search.best_score_),
            "Best_Parameters": search.best_params_
        })

        print("Best CV RMSE:", -search.best_score_)

    # ----------------------------------------------
    # 7. Select best model
    # ----------------------------------------------

    results_df = pd.DataFrame(results).sort_values(
        "CV_RMSE",
        ascending=True
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_cv_rmse = float(results_df.iloc[0]["CV_RMSE"])

    best_model = trained_models[best_model_name]

    print("\nBest model:", best_model_name)
    print("Best CV RMSE:", best_cv_rmse)

    # ----------------------------------------------
    # 8. Evaluate using test data
    # ----------------------------------------------

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model_name": best_model_name,
        "cv_rmse": best_cv_rmse,
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "features": X_train.columns.tolist(),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__
    }

    print("\nTest Evaluation:")
    print(json.dumps(metrics, indent=4))

    # ----------------------------------------------
    # 9. Save model artifacts
    # ----------------------------------------------

    model_path = MODEL_DIR / "best_superkart_model.joblib"

    joblib.dump(best_model, model_path)

    metrics_path = MODEL_DIR / "model_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save model comparison results
    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    # ----------------------------------------------
    # 10. Upload registered model to Hugging Face
    # ----------------------------------------------

    if os.getenv("REGISTER_MODEL", "false").lower() == "true":

        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is required to register the model"
            )

        api = HfApi(token=HF_TOKEN)

        # Upload trained model and evaluation artifacts
        api.upload_folder(
            folder_path=str(MODEL_DIR),
            repo_id=MODEL_REPO,
            repo_type="model",
            allow_patterns=[
                "best_superkart_model.joblib",
                "model_metrics.json",
                "model_comparison.csv"
            ],
            commit_message="Automated SuperKart model training"
        )

        print("Model registered successfully!")

    else:
        print(
            "Model registration skipped. "
            "Set REGISTER_MODEL=true to upload."
        )

    print("\nAutomated training completed successfully!")


if __name__ == "__main__":
    train_model()

