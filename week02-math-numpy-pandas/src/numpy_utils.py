import numpy as np

def normalize_features(X: np.ndarray) -> np.ndarray:
    """Z-score normalization: (x - mean) / std"""
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    return (X - mean) / (std + 1e-8)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Used in embeddings, RAG retrieval."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def softmax(x: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities - used in classification."""
    exp_x = np.exp(x - x.max())      # subtract max for numerical stability
    return exp_x / exp_x.sum()

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Split data - before you use sklearn's version."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]



# Practice
def min_max_scale(X: np.ndarray):
    """Scale an array X linearly to a range between 0 and 1."""
    min_val = np.min(X)
    max_val = np.max(X)

    return (X - min_val) / (max_val - min_val)

def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the cosine simimilarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / (norm_v1 * norm_v2)

# Testing 

# 1. Define vectors
v1 = np.array([10, 20, 30])
v2 = np.array([5, 15, 25])

# 2. Scale the vectors using min_max_scale
scaled_v1 = min_max_scale(v1)
scaled_v2 = min_max_scale(v2)

# 3. Compute the cosine similarity before and after scaling
before_sim = cos_sim(v1, v2)
after_sim = cos_sim(scaled_v1, scaled_v2)

print(f"Original Vector 1: {v1} -> Scaled: {scaled_v1}")
print(f"Original Vector 2: {v2} -> Scaled: {scaled_v2}")
print(f"Cosine Similarity (Before Scaling): {before_sim:.4f}")
print(f"Cosine Similarity (After Scaling): {after_sim:.4f}")
