import numpy as np

def preprocess_FERplus_impr(x):
    mean = [129.4432, 129.4432, 129.4432]
    std = [64.87448751, 64.87448751, 64.87448751]
    # ensure image format
    x = np.array(x, dtype='float32')
    # normalize
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x


def preprocess_RAF_impr(x):
    mean = [146.6770, 114.6274, 102.3102]
    std = [67.6282, 61.7651, 61.3665]
    # ensure image format
    x = np.array(x, dtype='float32')
    # normalize
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x
