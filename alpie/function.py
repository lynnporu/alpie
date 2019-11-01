"""Function classes.
"""


import inspect
import math
from copy import deepcopy


class Function():

    def __init__(self, fn):
        """Creates function from given callable object.
        """

        self.parameters = []
        self.fn = fn
        self.wrappers = []

        for name in inspect.signature(fn).parameters:
            self.parameters.append(name)

    def __hash__(self):
        return hash(self.fn, self.wrappers)

    def __repr__(self):
        return f"<Function params=({', '.join(self.parameters)})>"

    def __call__(self, *args, **kwargs):

        result = self.fn(*args, **kwargs)

        if self.wrappers:
            # TODO: implement functional wrappers
            for fn in self.wrappers:
                result = fn(result)

        return result

    def __deepcopy__(self, memdict):
        new = Function(fn=self.fn)
        new.wrappers = deepcopy(self.wrappers)
        return new

    @property
    def new(self):
        return deepcopy(self)

    def wrap(self, fn):
        self.wrappers.append(fn)
        return self

    def __len__(self):
        return len(self.parameters)

    def __lt__(self, other):
        return self.new.wrap(
            lambda n: n < other)

    def __le__(self, other):
        return self.new.wrap(
            lambda n: n <= other)

    def __eq__(self, other):
        return self.new.wrap(
            lambda n: n == other)

    def __ne__(self, other):
        return self.new.wrap(
            lambda n: n != other)

    def __gt__(self, other):
        return self.new.wrap(
            lambda n: n > other)

    def __ge__(self, other):
        return self.new.wrap(
            lambda n: n >= other)

    def __add__(self, other):
        return self.new.wrap(
            lambda n: n + other)

    def __radd__(self, other):
        return self.new.wrap(
            lambda n: other + n)

    # TODO: should +=, -=, ... methods modify self, or return new objects?

    def __sub__(self, other):
        return self.new.wrap(
            lambda n: n - other)

    def __rsub__(self, other):
        return self.new.wrap(
            lambda n: other - n)

    def __mul__(self, other):
        return self.new.wrap(
            lambda n: n * other)

    def __matmul__(self, other):
        return self.new.wrap(
            lambda n: n @ other)

    def __truediv__(self, other):
        return self.new.wrap(
            lambda n: n / other)

    def __floordiv__(self, other):
        return self.new.wrap(
            lambda n: n // other)

    def __mod__(self, other):
        return self.new.wrap(
            lambda n: n % other)

    def __divmod__(self, other):
        return self.new.wrap(
            lambda n: divmod(n, other))

    def __pow__(self, other):
        return self.new.wrap(
            lambda n: n ** other)

    def __rpow__(self, other):
        return self.new.wrap(
            lambda n: other ** n)

    def __lshift__(self, other):
        return self.new.wrap(
            lambda n: n << other)

    def __rlshift__(self, other):
        return self.new.wrap(
            lambda n: other << n)

    def __rshift__(self, other):
        return self.new.wrap(
            lambda n: n >> other)

    def __rrshift__(self, other):
        return self.new.wrap(
            lambda n: other >> n)

    def __and__(self, other):
        return self.new.wrap(
            lambda n: n & other)

    def __xor__(self, other):
        return self.new.wrap(
            lambda n: n ^ other)

    def __or__(self, other):
        return self.new.wrap(
            lambda n: n | other)

    def __neg__(self):
        return self.new.wrap(
            lambda n: -n)

    def __abs__(self):
        return self.new.wrap(
            lambda n: abs(n))

    def __invert__(self):
        return self.new.wrap(
            lambda n: ~n)

    def __round__(self, ndigits=None):
        return self.new.wrap(
            lambda n: round(n, ndigits))

    def __trunc__(self):
        return self.new.wrap(
            lambda n: math.trunc(n))

    def __floor__(self):
        return self.new.wrap(
            lambda n: math.floor(n))

    def __ceil__(self):
        return self.new.wrap(
            lambda n: math.ceil(n))
