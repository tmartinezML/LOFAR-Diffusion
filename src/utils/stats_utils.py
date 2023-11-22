import numpy as np


def norm(counts):
    return counts / np.sum(counts)


def norm_err(counts):
    return np.sqrt(counts / counts.sum()**2)


def cdf(counts):
    return norm(counts).cumsum(axis=0)


def cdf_err(counts):
    return np.sqrt(cdf(counts) / counts.sum())


def centers(edges):
    # Helper function to get bin centers from bin edges
    return (edges[1:] + edges[:-1]) / 2
