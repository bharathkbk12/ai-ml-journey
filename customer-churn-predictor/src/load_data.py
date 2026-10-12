from pathlib import Path
import pandas as pd

def load_churn_csv(path: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """Load Telco churn CSV; fail early if file missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Download CSV from Kaggle and place at {p}")
    df = pd.read_csv(p)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID, fix types, normalize target - no feature engineering yet."""
    df = df.copy()

    # customerID is not predictive; keeping it causes memorization in tree models
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # TotalCharges sometimes stored as string with spaces for new customers
    if "TotalCharges" in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    return df