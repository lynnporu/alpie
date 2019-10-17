"""This module includes methods for solving systems of linear equations,
"""


import matrices
import difference
import itertools
from copy import deepcopy
from functools import partial


def checktype(obj, needed):
    """Check type of object and raise error when not isinstance(needed). Use
    this function, when new instance can't be created from given primitive
    data (by `ensure` function).
    """
    if not isinstance(obj, needed):
        raise ValueError(f"{obj.__class__.__name__} needed for this function.")


class NotSymmetric(Exception):
    pass


def decompositionCholesky(matrix):
    """Perform Cholesky decomposition over the given SquareMatrix. Returns
    U and U-transposed.
    """

    matrix = matrices.SquareMatrix.ensure(matrix)
    if not matrix.isSymmetric:
        raise NotSymmetric("This method require symmetrical matrix.")

    n = len(matrix)

    T = matrices.SquareMatrix(size=n, data=0)

    T[0][0] = matrix[0][0] ** .5

    for i, j in itertools.product(range(0, n), repeat=2):

        if i == 0:
            T[i][j] = matrix[i][j] / T[i][i]

        elif i == j:
            S = 0
            for k in range(0, i):
                S += T[k][i] ** 2
            T[i][j] = (matrix[i][j] - S) ** .5

        elif i < j:
            S = 0
            for k in range(0, i):
                S += T[k][i] ** 2
            T[i][j] = (matrix[i][j] - S) / T[i][i]

        elif i > j:
            T[i][j] = 0

    return (T.transposed, T)


def gaussElimination(matrix, rowSorting=True):
    """Do Gaussian elimination over the given AugmentedMatrix.
    """
    checktype(matrix, matrices.AugmentedMatrix)

    new = deepcopy(matrix)
    sign = 1
    n = len(new)

    for i in range(0, n - 1):
        # Diagonal element.
        x = abs(new[i][i])
        maxRow = i

        if rowSorting:
            for k in range(i + 1, n):
                if abs(new[k][i]) > x:
                    x = abs(new[k][i])
                    maxRow = k

        if i != maxRow:
            new[i], new[maxRow] = new[maxRow], new[i]
            sign *= -1

        if abs(new[i][i]) < 1e-12:
            break

        # Gaussian elimination
        for k in range(i + 1, n):
            c = -new[k][i] / new[i][i]
            for j in range(i, n + 1):
                new[k][j] += c * new[i][j]

    return matrices.EliminatedAugmentedMatrix(
        matrices.TriangularMatrix(data=new.coeffs.data, sign=sign),
        matrix.eqs)


def gaussInversed(matrix):
    """Find inversed matrix by Gaussian elimination.
    """

    matrix = matrices.SquareMatrix.ensure(matrix)
    new = matrices.SquareMatrix.empty()

    for vector in matrix.identityMask.rows():
        new.data.append(
            gaussElimination(
                matrices.AugmentedMatrix(
                    coeffs=matrix,
                    eqs=matrices.Matrix.filledWith(vector).transposed))
            .roots)

    return new


def simpleIterationCoefficients(matrix):
    """Return matrices for simple fixed-point iteration.
    Returns tuple: matrices.SquareMatrix, matrices.Matrix.
    """
    checktype(matrix, matrices.AugmentedMatrix)

    A = matrices.SquareMatrix.empty()
    B = matrices.Matrix.empty()

    for aii, aij, bi in zip(
        matrix.coeffs.diagonal, matrix.coeffs.rows(), matrix.eqs.rows()
    ):
        A.data.append([-el / aii for el in aij])
        B.data.append([bi[0] / aii])

    A.zeroDiagonal()

    return (A, B)


def simpleIteration(matrix, initial=None, accuracy=1e-10, accuracyfunc=None):
    """Apply fixed-point iteration to this matrix with some initial
    approximation, which is instance of Matrix.
    accuracyfunc will be applied in order to find satisfying iteration
    (difference.simpleMax by default).
    """
    checktype(matrix, matrices.AugmentedMatrix)

    if initial:
        X = matrices.Matrix(initial)
        if X.dimensions != matrix.eqs.dimensions:
            raise matrices.InappropriateDimensions

    coeffsA, coeffsB = simpleIterationCoefficients(matrix)

    if not initial:
        X = deepcopy(coeffsB)

    return matrix.fixedPointIteration(
        X,
        partial(
            lambda C, d, x: d + C * x,
            coeffsA, coeffsB),
        partial(
            difference.simpleMax if not accuracyfunc else accuracyfunc,
            accuracy))


def seidelIteration(matrix, initial=None, accuracy=1e-10, accuracyfunc=None):
    """Apply Seidel iteration to given matrix with some initial approximation
    and given accuracy.
    accuracyfunc will be applied in order to find satisfying iteration
    (difference.simpleMax by default).
    """
    checktype(matrix, matrices.AugmentedMatrix)

    if initial:
        X = matrices.Matrix.ensure(initial)
        if X.dimensions != matrix.eqs.dimensions:
            raise matrices.InappropriateDimensions

    coeffsA, coeffsB = simpleIterationCoefficients(matrix)

    if not initial:
        X = deepcopy(coeffsB)

    def iterator(C, d, x):
        newX = matrices.Matrix.empty()
        # Calculate each equation apart
        for row, b in zip(C.rows(), d.elements()):
            newX.data.append([b + sum(
                [cn * xn for cn, xn in zip(row, x.elements())]
                # No x was calculated yet
                if not newX else
                [cn * newX[-1][0] for cn in row]
            )])
        return newX

    return matrix.fixedPointIteration(
        X,
        partial(iterator, coeffsA, coeffsB),
        partial(
            difference.simpleMax if not accuracyfunc else accuracyfunc,
            accuracy))
