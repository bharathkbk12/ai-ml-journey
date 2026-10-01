import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def build_pipeline(model=None) -> Pipeline:
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    return Pipeline([
        ('scalar', StandardScaler()),
        ('model', model)
    ])

def train_model(x, y, model=None, test_size=0.2, cv=5):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )
    pipe = build_pipeline(model)
    pipe.fit(x_train, y_train)

    results = {
        'train_accuracy': pipe.score(x_train, y_train),
        'test_accuracy': pipe.score(x_test, y_test),
        'cv_mean': cross_val_score(pipe, x_train, y_train, cv=cv).mean(),
        'Pipeline': pipe,
        'X_test': x_test,
        'Y_test': y_test,
    }
    return results

def save_model(pipe, path: str = 'week03-first-ml-model/models/titanic_model.pkl'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    print(f"Model Saved to {path}")

import seaborn as sns
import pandas as pd

# 1. Load the dataset from seaborn
titanic = sns.load_dataset('titanic')

# 2. Select numerical columns and drop rows with missing values
features = ['sex', 'embarked', 'age', 'sibsp', 'pclass', 'parch', 'fare']

# 3. Separate into features (x) and target (y)
x = titanic[features].copy()
y = titanic['survived'] # target variable

# 4. Fill the missing values
x['age'] = x['age'].fillna(x['age'].median())
x['embarked'] = x['embarked'].fillna(x['embarked'].mode()[0])

# 5. Encode Categorical Strings to Numbers
x = pd.get_dummies(x, columns=['sex', 'embarked'], drop_first=True)

# 6. Training the model
print("Training Titanic MOdel....")
test_results = train_model(x, y)

# 7. Display the results
print('-----Test results------')
print(f"Train Accuracy: {test_results['train_accuracy']:.3f}")
print(f"Test Accuracy: {test_results['test_accuracy']:.3f}")
print(f"CV Mean: {test_results['cv_mean']:.3f}")

# 8. Save Model
save_model(test_results['Pipeline'])