import numpy as np

def describe(arr: np.ndarray) -> dict:
    """Like pandas .describe() but from scratch."""
    return {
        "count": len(arr),
        "mean": arr.mean(),
        "std": arr.std(),
        "min": arr.min(),
        "25%": np.percentile(arr, 25),
        "50%": np.median(arr),
        "75%": np.percentile(arr, 75),
        "max": arr.max()
    }

def detect_outliers(arr: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Return boolean mask of outliers (> threshold std devs from mean)."""
    z_scores = np.abs((arr - arr.mean()) / arr.std())
    return z_scores > threshold
