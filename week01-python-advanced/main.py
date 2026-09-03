"""Week 1 Capstone: End-to-end mini ML pipeline using only Python stdlib."""

from oops.datasets import Dataset
from oops.models import MeanPredictor
from oops.config import TrainingConfig, TrainingResult
from oops.preprocessing import normalize, Pipeline
from decorators.timing import timer

@timer
def run_pipeline(config: TrainingConfig) -> TrainingResult:
    #1. Load data
    ds = Dataset(
        data = [[1.0], [2.0], [3.0], [4.0], [5.0]],
        labels = [2.0, 4.0, 6.0, 8.0, 10.0]
    )
    print(f"Loaded: {ds}")

    #2. Preprocess
    normalized_data = normalize(ds._data)

    #3. Train
    model = MeanPredictor()
    model.fit(normalized_data, ds._labels)
    print(f"Trained: {model}")

    #4. Predict
    predictions = model.predict([[0.5], [1.0]])
    print(f"Predictions: {predictions}")

    #5. Return results
    return TrainingResult(
        model_name = config.model_name,
        final_loss = 0.0,
        epochs_run = config.epochs,
        metrics = {"mean_prediction": predictions[0]}
    )

if __name__ == "__main__":
    config = TrainingConfig(epochs = 1, model_name = "MeanPredictor")
    result = run_pipeline(config)
    print(f"\nResult: {result}")

