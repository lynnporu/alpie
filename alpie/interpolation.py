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


def newton(xarr: list, yarr: list) -> function.Function:
    """Generate Newton's polynom interpolation function for given sets of
    x and y.
    """
    def pairs(iterable):
        """Generate overlapping pairs from iterable:
        ABCDEF -> AB BC CD DE EF
        """
        return zip(
            iter(iterable),
            itertools.islice(iter(iterable), 1, None))

    diffs = [deepcopy(yarr)]

    while len(diffs) < len(xarr):
        diffs.append([
            # TODO: work with generators
            (y1 - y0) / (xarr[i + len(diffs)] - xarr[i])
            for i, (y0, y1)
            in enumerate(pairs(diffs[-1]))
        ])

    def interpolation(x):
        result = 0
        for i, diff in enumerate([el[0] for el in diffs]):
            part = diff
            for mult in range(i):
                part *= x - xarr[mult]
            result += part
        return result

    return function.Function(interpolation)
