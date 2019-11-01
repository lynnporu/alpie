"""This module includes roots-finding alrorithms.
"""


import analysis


class BadApproximation(Exception):
    pass


def simpleIteration(function, a, b, accuracy=1e-6):
    """Apply step-by-step approximation method.
    """

    x = (a + b) / 2

    if a * b < 0:
        raise BadApproximation(
            "This approximation does not contain any root.")

    derivative = analysis.deltaDifferentiate(function)

    if function(x) * derivative(x):
        a, b = b, a

    while True:

        x = b - (function(b) / derivative(b))

        if abs(x - b) < accuracy:
            b = x

        else:
            break

    return x
