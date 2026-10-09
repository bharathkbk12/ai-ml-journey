# Titanic Survival Predictor

## Problem
Predict whether a passenger survived the Titanic disaster.

## Approach
1. Cleaned data (handled 20% missing ages, dropped leaky features)
2. Engineered 5 new features (family_size, is_alone, fare_per_person, etc.)
3. Trained Random Forest with 5-fold cross-validation
4. Tuned hyperparameters with GridSearchCV

## Results
| Metric | Score |
|--------|-------|
| Accuracy | ~0.82 |
| F1 Score | ~0.78 |
| ROC-AUC | ~0.85 |

## Key Findings
- Sex is the strongest predictor (female survival rate: 74%)
- Family size affects survival (small families do better)
- Fare per person correlates with survival

## Project Structure
- src/features.py — data cleaning + feature engineering
- src/train.py — training pipeline
- src/evaluate.py — metrics
- notebooks/ — exploratory analysis