"""Physical spaces.
"""


import itertools
import functools
import operator
import math


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

    def __len__(self):
        """Get number of discrete points in this figure.
        """
        return self.size / self.detalization

    def __iter__(self):
        raise NotImplemented

    def __contains__(self):
        raise NotImplemented

    def size(self):
        raise NotImplemented


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

    def size(self):
        """Get continuum size of figure.
        """
        return functools.reduce(
            # Multiply sizes
            operator.mul, [
                # Calculate sizes of the figure.
                (a - b)
                for a, b
                in zip(self.start, self.end)])


class Square(Rectangle):
    """A multidimensional square.
    """

    def __init__(self, start: tuple, size: float, detalization=1e-3):
        """Creates square with coordinates:
        (start[0], start[1], ..., start[dim]) ->
            (start[0] + size, start[1] + size, ..., start[dim] + size)
        """
        self.start = start
        self.end = tuple(x + size for x in start)
        self.detalization = detalization


class Range(Space):
    """A simple one-dimensional range of space.
    """

    def __init__(self, start: float, end: float, detalization=1e-3):
        """Creates range with coordinates:
        (start) -> (end)
        """
        self.start = start
        self.end = end
        self.detalization = detalization

    def __iter__(self):
        return frange(self.start, self.end, self.detalization)

    def size(self):
        """Continuum size of the range.
        """
        return abs(self.start - self.end)


class ClosedHyperball(Space):
    """A multidimensional closed ball.
    """

    def __init__(self, center: tuple, radius: float, detalization=1e-3):
        """Creates ball with center in given coordinates. Your ball will have
        dimension equal to len(center).
        """
        self.start = center
        self.radius = radius
        self.detalization = detalization

    def __iter__(self):
        return [
            point
            for point
            # TODO: Not efficient. Replace with some trigonometry.
            in Square(self.start, self.radius, self.detalization)
            if point in self
        ]

    def __contains__(self, point: tuple):
        return sum(x ** 2 for x in point) <= self.radius ** 2

    def size(self):
        """Continuum size of your hyperball.
        """
        return (
            (math.pi ** (len(self.size) / 2)) / math.gamma(len(self.size) + 1)
        ) * self.radius ** len(self.size)


class OpenedHyperball(Space):
    """A multidimensional opened ball.
    """

    def __contains__(self, point: tuple):
        return sum(x ** 2 for x in point) < self.radius ** 2
