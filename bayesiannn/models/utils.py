import numpy as np

# import math


def relu(x: float):
    return np.maximum(0, x)


def softmax(x: np.ndarray):
    e_x: np.ndarray = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def identity(x: np.ndarray):
    return x


def entropy(x, base: int = 2):
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.emath.logn(base, probs))
