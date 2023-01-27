import numpy as np


def vectorized_correlation(x, y, reduction="mean"):
    dim = 0
    x = np.array(x)
    y = np.array(y)

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    if reduction is None:
        return corr.ravel()
    elif reduction == "mean":
        return corr.ravel().mean()
    else:
        raise Exception("Unknown reduction")

