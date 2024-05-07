import numpy as np
from astropy.stats import poisson_conf_interval


def norm(counts):
    return counts / np.sum(counts)


def norm_err(counts):
    return np.sqrt(counts) / np.sum(counts)


def norm_err_poisson(counts):
    CI = poisson_conf_interval(counts, interval="frequentist-confidence")
    err = [
        (counts - CI[0]) / np.sum(counts),
        (CI[1] - counts) / np.sum(counts),
    ]

    # Set to 0 if counts are 0
    err[0][counts == 0] = 0
    err[1][counts == 0] = 0

    return err


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

    def W1_err_term(c):
        return np.sum(cdf(c)) / np.sum(c)  # helper func.

    W1_err = np.sqrt(W1_err_term(C1) + W1_err_term(C2))
    return W1.item(), W1_err.item()
