import pandas as pd 
import numpy as np
import seaborn as sns

def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and Engineer features for Titanic"""
    df = df.copy()
    # Create cabin availability feature before dropping the cabin column
    df['has_cabin'] = df['deck'].notna().astype(int)
    df = df.drop(columns=['deck', 'embark_town', 'alive', 'class'], errors='ignore')
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = df.dropna(subset=['fare'])
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features."""
    df = df.copy()
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['fare_per_person'] = df['fare'] / df['family_size']
    df['sex_encoded'] = (df['sex'] == 'female').astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['child', 'teen', 'young_adult', 'adult', 'senior'])
    df = pd.get_dummies(df, columns=['embarked', 'age_group'], drop_first=True)
    return df

def get_feature_matrix(df: pd.DataFrame, target_col: str = 'survived'):
    """Return x, y ready for sklearn."""
    drop_cols = [target_col, 'who', 'adult_male']
    x = df.drop(columns=[c for c in drop_cols if c in df.columns])
    x = x.select_dtypes(include='number')
    y = df[target_col]
    return x, y


# 1. Load Dataset
data = sns.load_dataset('titanic')

print("Original Shape:", data.shape)

# 2. Clean data
data = clean_titanic(data)

print("After cleaning:", data.shape)

# 3. Create Features
data = engineer_features(data)

print("Feature Engineering:", data.shape)

# 4. Create x and y
x, y = get_feature_matrix(data)

print("X shape:", x.shape)
print("Y shape:", y.shape)

print("Features:")
print(x.columns.tolist())

print("First 5 rows:")
print(x.head())

print("Target:")
print(y.head())