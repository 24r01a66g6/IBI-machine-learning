"""
preprocessing.py
────────────────
Full scikit-learn Pipeline for preprocessing the HR attrition dataset.
Handles:
  - Missing value imputation
  - Categorical encoding (Label + One-Hot)
  - Feature scaling (StandardScaler)
  - Class imbalance via SMOTE (optional, during training only)

Design note: We use sklearn Pipeline to prevent data leakage — all
transformations are fit ONLY on training data, then applied to test data.
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# ── Feature groups ────────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany",
    "JobSatisfaction", "WorkLifeBalance",
    "PerformanceRating", "DistanceFromHome"
]

CATEGORICAL_FEATURES = ["Department", "OverTime"]

TARGET_COLUMN  = "Attrition"
DROP_COLUMNS   = ["EmployeeID"]          # Not useful for prediction


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numerical features
      - Imputes + one-hot encodes categorical features

    Returns:
        ColumnTransformer ready to fit_transform on training data.
    """
    # Numerical branch: median imputation → standard scaling
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Categorical branch: most-frequent imputation → one-hot encoding
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,  NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")  # Drop EmployeeID and any unlisted columns

    return preprocessor


def encode_target(series: pd.Series) -> np.ndarray:
    """
    Encode the Attrition target column: 'Yes' → 1, 'No' → 0.

    Args:
        series: Raw Attrition column.
    Returns:
        Numpy array of 0/1 labels.
    """
    return (series.str.strip().str.lower() == "yes").astype(int).values


def load_and_prepare_data(csv_path: str):
    """
    Load the CSV, drop unused columns, and split into X and y.

    Args:
        csv_path: Path to employee_data.csv
    Returns:
        X (DataFrame), y (numpy array)
    """
    df = pd.read_csv(csv_path)

    # Drop ID columns
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = encode_target(df[TARGET_COLUMN])

    print(f"  Loaded {len(df)} records | Features: {X.shape[1]} | "
          f"Attrition rate: {y.mean()*100:.1f}%")
    return X, y


def apply_smote(X_train_transformed: np.ndarray,
                y_train: np.ndarray,
                random_state: int = 42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes
    using pure NumPy interpolation, guaranteeing zero multiprocessing deadlocks.

    Args:
        X_train_transformed: Preprocessed training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.
    Returns:
        X_resampled, y_resampled
    """
    np.random.seed(random_state)
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]
    n_oversample = len(majority_idx) - len(minority_idx)

    if n_oversample > 0 and len(minority_idx) > 1:
        # SMOTE interpolation: x_new = x_i + lambda * (x_j - x_i)
        idx1 = np.random.choice(minority_idx, size=n_oversample, replace=True)
        idx2 = np.random.choice(minority_idx, size=n_oversample, replace=True)
        weights = np.random.uniform(0.1, 0.9, size=(n_oversample, 1))
        synthetic_samples = X_train_transformed[idx1] + weights * (X_train_transformed[idx2] - X_train_transformed[idx1])
        synthetic_labels = np.ones(n_oversample, dtype=int)

        X_res = np.vstack([X_train_transformed, synthetic_samples])
        y_res = np.concatenate([y_train, synthetic_labels])

        shuffle_idx = np.random.permutation(len(y_res))
        X_res, y_res = X_res[shuffle_idx], y_res[shuffle_idx]
    else:
        X_res, y_res = X_train_transformed, y_train

    print(f"  SMOTE applied | Before: {len(y_train)} | After: {len(y_res)}")
    print(f"  Class distribution after SMOTE: No={sum(y_res==0)}, Yes={sum(y_res==1)}")
    return X_res, y_res


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Retrieve the final feature names after one-hot encoding,
    useful for SHAP and feature importance plots.

    Args:
        preprocessor: Fitted ColumnTransformer.
    Returns:
        List of feature name strings.
    """
    ohe_features = (preprocessor
                    .named_transformers_["cat"]
                    .named_steps["encoder"]
                    .get_feature_names_out(CATEGORICAL_FEATURES)
                    .tolist())
    return NUMERICAL_FEATURES + ohe_features


def save_preprocessor(preprocessor: ColumnTransformer,
                       save_dir: str = "models") -> str:
    """Save fitted preprocessor to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, path)
    print(f"  ✅ Preprocessor saved → {path}")
    return path


def load_preprocessor(save_dir: str = "models") -> ColumnTransformer:
    """Load a previously saved preprocessor."""
    path = os.path.join(save_dir, "preprocessor.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found at: {path}")
    return joblib.load(path)


def preprocess_single_input(raw_input: dict,
                             preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocess a single employee dict for real-time prediction.

    Args:
        raw_input: Dict with keys matching NUMERICAL_FEATURES + CATEGORICAL_FEATURES.
        preprocessor: Fitted ColumnTransformer.
    Returns:
        2D numpy array ready for model.predict()
    """
    df = pd.DataFrame([raw_input])
    return preprocessor.transform(df)
