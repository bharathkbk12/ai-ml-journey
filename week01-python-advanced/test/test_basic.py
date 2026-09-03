import pytest
from oops.datasets import Dataset
from oops.models import MeanPredictor
from oops.config import TrainingConfig

def test_dataset_length():
    ds = Dataset([1, 2, 3], [0, 1, 0])
    assert len(ds) == 3

def test_dataset_getitem():
    ds = Dataset([1, 2], [10, 20])
    assert ds[0] == (1, 10)

def test_model_fit_predict():
    model = MeanPredictor()
    model.fit([[1], [2], [3]], [10, 20, 30])
    preds = model.predict([[4]])
    assert preds == [20.0]

def test_config_validation():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate =- 1)

def test_unfitted_model_raises():
    model = MeanPredictor()
    with pytest.raises(RuntimeError):
        model.predict([[1]])

