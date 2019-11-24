import function
import functools
import itertools
import operator
from copy import deepcopy


def lagrange(xarr: list, yarr: list) -> function.Function:
    """Generate Lagrange polynomial interpolation function for given sets of x
    and y.
    """
    def mult(iterable):
        """Multiply all elements of iterable.
        """
        return functools.reduce(operator.mul, iterable)

    def interpolation(x):
        return sum([
            yi * (
                mult(
                    x - xj for j, xj in enumerate(xarr)
                    if i != j
                ) /
                mult(
                    xi - xj for j, xj in enumerate(xarr)
                    if i != j
                )
            )
            for xi, yi, i
            in zip(xarr, yarr, range(len(xarr)))
        ])

    return function.Function(interpolation)
