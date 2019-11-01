"""Physical spaces.
"""


import itertools
import functools
import operator


def frange(start, stop, step):
    """Float range generator.
    """
    count = 0
    result = start
    while result <= stop:
        yield result
        count += 1
        result = start + (step * count)


class Space():
    pass


class Rectangle(Space):
    """A multidimensional rectangle.
    """

    def __init__(self, start: tuple, end: tuple, detalization=1e-3):
        """Creates rectangle with coordinates:
        (start[0], start[1], ...) -> (end[0], end[1], ...)
        """
        self.start = start
        self.end = end
        self.detalization = detalization

    def __iter__(self):
        return itertools.product(*[
            frange(a, b, self.detalization)
            for a, b
            in zip(self.start, self.end)])

    def __len__(self):
        """Get number of discrete points in this figure.
        """
        return int(
            functools.reduce(
                operator.mul,
                [
                    a - b
                    for a, b
                    in zip(self.end, self.start)
                ]
            ) / self.detalization
        )


class Square(Rectangle):
    """A multidimensional square.
    """

    def __init__(self, start: tuple, size: int, detalization=1e-3):
        """Creates square with coordinates:
        (start[0], start[1], ..., start[dim]) ->
            (start[0] + size, start[1] + size, ..., start[dim] + size)
        """
        self.start = start
        self.end = tuple(x + size for x in start)
        self.detalization = detalization
