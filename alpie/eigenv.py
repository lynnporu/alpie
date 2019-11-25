"""This module include methods for finding eigenvalues and eigenvectors.
"""

import matrices
import lineqs
import functools
import operator
import math


class WrongInitial(Exception):
	pass


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
    matrix = matrices.SquareMatrix.ensure(matrix)
    initial = matrices.Matrix.ensure(initial)

    if not solvingfunc:
        def solvingfunc(A, b):
            return lineqs.gaussElimination(
                matrices.AugmentedMatrix(
                    coeffs=A, eqs=b)).roots

    prevX = initial

    while num > 0:
        yk = solvingfunc(
            (matrix - matrix.identityMask * rayleighQuotient(matrix, prevX)),
            prevX)
        prevX = yk / yk.euclideanNorm
        num -=1

    return prevX


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
        matrix = H.transposed @ matrix @ H

    return zip(
        # Make eigenpairs (eigenvalue, eigenvector).
        matrix.diagonal,
        # Multiply all H-matrices and get its rows.
        functools.reduce(
            operator.matmul, hmatrices).rows()
    )

