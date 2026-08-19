#Test oops/datasets.py
from oops.datasets import Dataset

ds = Dataset([1, 2, 3], [0, 1, 0])
print(len(ds))       # 3
print(ds[0])         # (1, 0)
print(ds.shape)      # (3,)

#Test oops/models.py

from oops.datasets import Dataset
from oops.models import MeanPredictor

ds = Dataset([[1], [2], [3]], [10, 20, 30])
model = MeanPredictor()
model.fit(ds._data, ds._labels)
print(model.predict([[4], [5]])) #[20.0, 20.0]
print(model) #MeanPredictor(fitted)