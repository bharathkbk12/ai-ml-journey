import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    return metrics

def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    metrics = evaluate_model(y_true, y_pred)
    for k, v in metrics.items():
        print(f" {k}: {v:.3f}")

# Load dataset for testing
df = sns.load_dataset('titanic')

features = ['pclass', 'sibsp', 'age', 'parch', 'fare', 'sex', 'embarked']

x = df[features]
y = df['survived']

x['age'] = x['age'].fillna(x['age'].median())
x['fare'] = x['fare'].fillna(x['fare'].median())
x['embarked'] = x['embarked'].fillna(x['embarked'].mode()[0])
x['sex'] = x['sex'].map({'male': 0, 'female': 1})

x = pd.get_dummies(x['embarked'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_true = y_test
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test) [:, 1]

print("-------Classification Report-------")
print_report(y_true, y_pred)

print("-------Evaluation Metrics-------")
metrics = evaluate_model(y_true, y_pred, y_prob)
for k, v in metrics.items():
    print(f" {k}: {v:.3f}")