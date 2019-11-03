"""This module includes roots-finding alrorithms.
"""


import function
import physical
from copy import deepcopy


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

    def phi(x):
        return x - m * f(x)

    if f(a) * f(b) > 0:
        raise BadApproximation(
            "This approximation does not contain any root.")

    x = initial

    while True:

        xn = phi(x)
        diff = xn - x
        x = xn

        if abs(diff) < accuracy:
            break

    return x


def gradientDescent(
    executable, variables: list, initial: tuple, accuracy=1e-6
):
    """Apply gradient descent method to find roots of given function.
    `executable` may be instance of function.RnFunction or
    function.FunctionalVector. If `executable` is vector, then gradient
    method will be applied to new (fn ** 2 for fn in executable) function.
    `variables` is list of names,
    `initial` is tuple of coordinates of initial point.
    """

    phi = (
        executable.sqrsum
        if isinstance(executable, function.FunctionalVector) else
        executable
    )
    phigr = -phi.grad(variables)
    initial = function.ScalarVector.ensure(initial)

    def step(prev, vector, alpha=1):
        """Calculates next step of descent. Here:
        `prev` is phi value with previous vector,
        `vector` is previous vector.
        New vector will be returned as the result.
        """

        alpha = 1
        calc = None

        while True:

            # Right arithmetic operators are not supported for ScalarVectors
            calc = phigr(**vector.ofNames(variables)) * alpha + vector
            new = phi(**calc.ofNames(variables))

            if new == 0:
                break

            if new >= prev:
                alpha /= 2
            else:
                break

        return calc

    while True:

        calc = step(
            phi(**initial.ofNames(variables)), initial)

        if max(abs(calc - initial)) < accuracy:
            break

        initial = deepcopy(calc)

    return initial
