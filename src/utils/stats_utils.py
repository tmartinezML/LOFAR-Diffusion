import numpy as np


def norm(counts):
    return counts / np.sum(counts)


def norm_err(counts):
    return np.sqrt(counts) / np.sum(counts)


def cdf(counts):
    return norm(counts).cumsum(axis=0)


def cdf_err(counts):
    return np.sqrt(cdf(counts) / counts.sum())


def centers(edges):
    # Helper function to get bin centers from bin edges
    return (edges[1:] + edges[:-1]) / 2

def W1_distance(C1, C2):
    # Calculate W1 & error
    W1 = np.abs(cdf(C1) - cdf(C2)).sum()
    def W1_err_term(c): return np.sum(cdf(c)) / np.sum(c)  # helper func.
    W1_err = np.sqrt(W1_err_term(C1) + W1_err_term(C2))
    return W1.item(), W1_err.item()