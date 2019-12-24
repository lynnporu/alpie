import function
import functools
import itertools
import operator
import physical
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


def bezierf(obj1, obj2):
    """Returns single parameter function "t", which provides bezier
    interpolation between two objects. It might be two points or two functions
    from previous bezier interpolations.
    """

    def interpolator(t):

        return (
            (obj1 if isinstance(obj1, physical.Point) else obj1(t)) * t +
            (obj2 if isinstance(obj2, physical.Point) else obj2(t)) * (1 - t)
        )

    return function.Function(interpolator)
