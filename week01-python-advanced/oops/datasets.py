"""A simple Dataset class — mimics PyTorch/sklearn patterns."""

class Dataset:
    def __init__(self, data: list, labels: list):
        if len(data) != len(labels):
            raise ValueError("data and labels must be same length")
        self._data = data
        self._labels = labels

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index], self._labels[index]

    def __repr__(self) -> str:
        return f"Dataset(samples={len(self)})"

    @property
    def shape(self) -> tuple:
        return (len(self._data),)

    @classmethod
    def from_csv(cls, filepath: str) -> "Dataset":
        """Factory method — you'll implement this Thursday."""
        raise NotImplementedError("Coming on Thursday")

    @staticmethod
    def validate(data: list, labels: list) -> bool:
        return len(data) == len(labels) and len(data) > 0

