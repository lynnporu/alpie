"""This module includes functions for finding eigenvalues and eigenvectors.
"""

import matrices


def rayleighFunc(matrix, vector):
    matrix = matrices.Matrix.ensure(matrix)
    vector = matrices.Matrix.ensure(vector)

    return (matrix @ vector) * vector / \
        vector * vector


def powerIteration(matrix, initial, num):
    """Perform power iteration method in order to find eigenvalue of given
    matrix.
    """
    matrix = matrices.SquareMatrix.ensure(matrix)
    vector = matrices.Matrix.ensure(initial)

    while num > 0:
        vector = (matrix @ vector) / vector.euclideanNorm
        num -= 1

    return vector
