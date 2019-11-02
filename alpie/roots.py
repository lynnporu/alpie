"""This module includes roots-finding alrorithms.
"""


import function
import physical


class BadApproximation(Exception):
    pass


def simpleIteration(
    f: function.Function, a, b, initial, accuracy=1e-6
):
    """Apply step-by-step approximation method to find roots of the following
    expression:
    f(...) = 0
    """

    f = function.Function.ensure(f)

    m = 1 / f.derivative().findByComparison(
        physical.Range(a, b, detalization=1e-2), max)

    newfunc = lambda x: x - m * f(x)

    if f(a) * f(b) > 0:
        raise BadApproximation(
            "This approximation does not contain any root.")

    x = initial

    while True:

        xn = newfunc(x)
        diff = xn - x
        x = xn

        if abs(diff) < accuracy:
            break

    return x
