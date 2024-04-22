import numpy as np
from matplotlib import pyplot as plt
import math 
import time

def ith_largest_singular_value(M, i):
    """
    Find the i-th largest singular value of the matrix M.

    Parameters:
    - M: A numpy array representing the matrix whose singular values we want to compute.
    - i: The index for the singular value (1-based; 1 is the largest singular value).

    Returns:
    - The i-th largest singular value of M.
    """
    # Compute the singular values using SVD
    singular_values = np.linalg.svd(M, compute_uv=False)
    # Return the i-th largest singular value
    # Since the singular values are returned in descending order, the index i needs to be adjusted by -1.
    return singular_values[i - 1]

def sample_from_ball(d, radius):
    # U^{\frac{1}{d}} \times \frac{Y}{||Y||}

    Y = np.random.randn(d)
    U = np.random.uniform(0, 1)
    # Scale Y by (U^(1/d)) / ||Y|| to get a uniformly sampled point from the ball
    point = radius * (U ** (1/d)) * Y / np.linalg.norm(Y)
    print("point here!:", point)
    
    return point