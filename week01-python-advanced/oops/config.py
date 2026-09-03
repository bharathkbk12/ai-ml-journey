# oop/config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    model_name: str = "MeanPredictor"
    random_seed: Optional[int] = 42

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

@dataclass
class TrainingResult:
    model_name: str
    final_loss: float
    epochs_run: int
    metrics: dict[str, float] = field(default_factory=dict)

import csv
from pathlib import Path
from oops.datasets import Dataset

@classmethod
def from_csv(cls, filepath: str, label_col: str = "label") -> "Dataset":
    path = Path(filepath)
    data, labels = [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(float(row.pop(label_col)))
            data.append([float(v) for v in row.values()])
    return cls(data, labels)