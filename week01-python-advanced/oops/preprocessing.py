from typing import Callable

def normalize(data: list[list[float]]) -> list[list[float]]:
    """Min-max normalize each column."""
    if not data:
        return data
    n_cols = len(data[0])
    mins = [min(row[i] for row in data) for i in range(n_cols)]
    maxs = [max(row[i] for row in data) for i in range(n_cols)]
    return [
        [(row[i] - mins[i]) / (maxs[i] - mins[i] + 1e-8) for i in range(n_cols)]
        for row in data
    ]

def filter_outliers(data: list, labels: list, threshold: float = 3.0) -> tuple:
    """Remove rows where any feature exceeds threshold std devs."""
    import statistics
    means = [statistics.mean(col) for col in zip(*data)]
    stds = [statistics.stdev(col) for col in zip(*data)]

    filtered = [
        (row, label) for row, label in zip(data, labels)
        if all(abs(r - m) <= threshold * s for r, m, s in zip(row, means, stds))
    ]
    if not filtered:
        return data, labels
    new_data, new_labels = zip(*filtered)
    return list(new_data), list(new_labels)

class Pipeline:
    """Chain preprocessing steps - like sklearn Pipeline."""

    def __init__(self, steps: list[tuple[str, Callable]]):
        self.steps = steps

    def transform(self, data, labels=None):
        for name, func in self.steps:
            if labels is not None:
                data, labels = func(data, labels)
            else:
                data = func(data)
        return (data, labels) if labels is not None else data

