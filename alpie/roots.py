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


def systemGradientDescent(
    fns: function.FunctionalVector, variables: list, initial: tuple,
    accuracy=1e-6
):
    """Apply gradient method to given function vector.
    `variables` is list of names.
    """

    phi = fns.sqrsum
    phigr = -phi.grad(variables)
    initial = function.ScalarVector.ensure(initial)

    def step(prev, vector, alpha=1):
        """Calculates next step of descent,
        `prev` is phi value with previous vector,
        `vector` is previous vector.
        New vector will be returned as the result.
        """
        # Right arithmetic operators are not supported for ScalarVectors
        calc = phigr(**vector.ofNames(variables)) * alpha + vector
        new = phi(**calc.ofNames(variables))

        if new >= prev:
            return step(prev, vector, alpha / 2)
        else:
            return calc

    while True:

        calc = step(
            phi(**initial.ofNames(variables)), initial)

        print(calc)

        if max(abs(calc - initial)) < accuracy:
            break

        initial = deepcopy(calc)

    return initial
