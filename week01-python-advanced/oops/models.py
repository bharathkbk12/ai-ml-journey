from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """Abstract Base - like sklearn BaseEstimator"""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, Y) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X) -> list:
        pass

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "notfitted"
        return f"{self.__class__.__name__}({status})"


class MeanPredictor(BaseModel):
    """Predicts the mean of the training labels - simplest ML model."""

    def __init__(self):
        super().__init__()
        self. _mean: float = 0.0

    def fit(self, X, Y) -> "MeanPredictor":
        self._mean = sum(Y) / len(Y)
        self.is_fitted = True
        return self

    def predict(self, X) -> list:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return [self._mean] * len(X)

