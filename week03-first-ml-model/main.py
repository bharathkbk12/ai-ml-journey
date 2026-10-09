"""
Titanic Survival Predictor - Week 3 Capstone
End-to-end: load -> clean -> engineer -> train -> evaluate -> save
"""

import seaborn as sns
from src.features import clean_titanic, engineer_features, get_feature_matrix
from src.train import train_model, save_model
from src.evaluate import evaluate_model, print_report

def main():
    # 1. Load and prepare
    print("Loading data...")
    df = sns.load_dataset('titanic')
    df = clean_titanic(df)
    df = engineer_features(df)
    X, y = get_feature_matrix(df)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Train
    print("\n Training Random Forest...")
    results = train_model(X, y)
    pipe = results['Pipeline']
    print(f" Train accuracy: {results['train_accuracy']:.3f}")
    print(f" Test accuracy: {results['test_accuracy']:.3f}")
    print(f" CV mean: {results['cv_mean']:.3f}")

    # 3. Evaluate
    print("\n Detailed evaluation:")
    y_pred = pipe.predict(results['X_test'])
    y_prob = pipe.predict_proba((results['X_test']))[:, 1]
    print_report(results['Y_test'], y_pred)
    metrics = evaluate_model(results['Y_test'], y_pred, y_prob)
    print(f" ROC-AUC: {metrics['roc_auc']:.3f}")

    # 4. Save
    save_model(pipe)
    print("\n Done: Model saved to models/titanic_model.pkl")


if __name__ == '__main__':
    main()

