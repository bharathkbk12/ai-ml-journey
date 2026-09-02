import time
import functools
from typing import Callable

def timer(func: Callable) -> Callable:
    """Decorator that prints execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def validate_input(func: Callable) -> Callable:
    """Decorator that validates data is not empty."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and len(args[0]) == 0:
            raise ValueError("Input data cannot be empty")
        return func(*args, **kwargs)
    return wrapper

def memoize(func: Callable) -> Callable:
    """Simple caching decorator."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper


from decorators.timing import timer, validate_input
from oops.models import BaseModel

class MeanPredictor(BaseModel):
    @timer
    @validate_input
    def fit(self, X, y) -> "MeanPredictor":
        self._mean = sum(y) / len(y)
        self.is_fitted = True
        return self