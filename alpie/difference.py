"""This module includes functions for evaluating differences between matrices.
"""


import matrices
import operator


def simpleMax(eps, a, b):
    """Find difference between two given matrices by this formula:
    max(a[i] - b[i]), where a[i] is i-th coordinate of matrix,
    and check if it is <= than given eps.
    """
    a = matrices.Matrix.ensure(a)
    b = matrices.Matrix.ensure(b)
    return max(map(
        operator.sub, a.elements(), b.elements())) <= eps
