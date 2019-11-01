"""Functions for derivatives and anti-derivatives.
"""


from functools import partial


def deltaDifferentiate(function):

    def derivative(fn, x, delta=1e-12):
        return (fn(x + delta) - fn(x)) / delta

    return partial(derivative, function)
