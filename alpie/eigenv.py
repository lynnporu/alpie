"""This module include methods for finding eigenvalues and eigenvectors.
"""

import matrices
import functools
import operator
import math
from lineqs import gaussElimination


class WrongInitial(Exception):
    pass


def rayleighQuotient(matrix, vector):
    """Calculates Rayleigh quotient for given matrix and vector.
    """

    matrix = matrices.Matrix.ensure(matrix)
    vector = matrices.Matrix.ensure(vector)

    return (
    (matrix * vector).scalarMul(vector) / \
    vector.scalarMul(vector))


def rayleighIteration(matrix, initial, num):
    """Finds eigenvalue by Rayleigh iteration with given initial matrix.
    """

    matrix = matrices.SquareMatrix.ensure(matrix)
    initial = matrices.Matrix.ensure(initial)

    if initial.euclideanNorm != 1:
        raise WrongInitial(
            "This method require initial matrix norm to be equal to 1.")

    for _ in range(num):
        yk = matrices.Matrix.ensure(gaussElimination(
            matrices.AugmentedMatrix(
                coeffs=(
                    matrix -
                    matrix.identityMask * rayleighQuotient(
                        matrix, initial)),
                eqs=initial
            )).roots).transposed
        initial = yk / yk.euclideanNorm

    return initial


def jacobi(matrix, accuracy=1e-5):
    """Finds all eigenpairs by Jacobi method.
    """

    matrix = matrices.SquareMatrix.ensure(matrix)
    if not matrix.isSymmetric:
        raise ValueError("This method requires symmetric matrix.")

    hmatrices = list()

    def findMax():
        """Find maximal element and its coordinates
        """
        found = (0, 0, 0)
        # enum() method returns (element, coordinates), where
        # `coordinates` is a tuple of i, j for SquareMatrix
        for (element, (i, j)) in matrix.enum:
            if element > found[0] and i != j:
                # Note, that `found` and `item` has different structure
                found = (element, i, j)
        return found

    while True:

        maximal, i, j = findMax()
        if maximal <= accuracy:
            break

        try:
            theta = .5 * math.atan(
                (2 * maximal) / (matrix[i][i] - matrix[j][j]))
        except ZeroDivisionError:
            theta = math.pi / 4

        H = matrix.givensMask(i, j, theta)
        hmatrices.append(H)
        matrix = H.transposed * matrix * H

    return zip(
        # Make eigenpairs (eigenvalue, eigenvector).
        matrix.diagonal,
        # Multiply all H-matrices and get its rows.
        functools.reduce(
            operator.mul, hmatrices).rows()
    )
