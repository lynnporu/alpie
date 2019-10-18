"""This module includes functions for finding eigenvalues and eigenvectors.
"""

import lineqs
import matrices


def rayleighQuotient(matrix, vector):
    matrix = matrices.Matrix.ensure(matrix)
    vector = matrices.Matrix.ensure(vector)

    return ((matrix @ vector) * vector) / vector ** 2


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

def RQIteration(matrix, initial, num, solvingfunc=None):
    """Apply Rayleigh quotient iteration to the given matrix.
    solvingfunc will be used in order to solve SLAE (Gaussian elimination
    by default).
    """
    matrix = matrices.Matrix.ensure(matrix)
    initial = matrices.Matrix.ensure(initial)

    if not solvingfunc:
        def solvingfunc(A, b):
            return lineqs.gaussElimination(
                matrices.AugmentedMatrix(
                    coeffs=A, eqs=b)).roots

    prevX = initial

    while num > 0:
        yk = solvingfunc(
            (
                matrix - \
                matrices.SquareMatrix.ofIdentity(len(matrix)) * \
                rayleighQuotient(matrix, prevX)
            ),
            prevX)
        prevX = yk / yk.euclideanNorm
        num -=1

    return prevX
