"""Function classes.
"""


import inspect
import functools
import operator
import math
from copy import deepcopy


class Executable():
    """Wrapper for built-in functions.
    """

    def __init__(self, fn):
        """Creates function from given callable object.
        """
        self.parameters = []
        self.executable = fn

        for name in inspect.signature(fn).parameters:
            self.parameters.append(name)

    def __hash__(self):
        return hash(self.fn)

    def __repr__(self):
        return f"<Executable params=({', '.join(self.parameters)})>"

    def __call__(self, **kwargs):

        if not all([name in kwargs for name in self.parameters]):
            raise TypeError(
                "One of these parameters is not present: "
                f"{', '.join(self.parameters)}")

        return self.executable(
            # Filter variables function doesn't accept and populate it to
            # arguments.
            **{
                name: value
                for (name, value)
                in kwargs.items()
                if name in self.parameters})


class Function():

    def __init__(self, fn):
        """Creates function from given callable object.
        """
        self.core = Executable(fn)
        self.wrappers = []

    def __repr__(self):
        return "<{} core with{}wrappers>".format(
            self.core.__repr__(),
            " no " if not self.wrappers else " ")

    def __hash__(self):
        return hash(self.executable, self.wrappers)

    def __call__(self, **kwargs):

        result = self.core(**kwargs)

        for fn in self.wrappers:
            result = fn(result, **kwargs)

        return result

    def __deepcopy__(self, memdict):
        new = Function(fn=self.core.executable)
        new.wrappers = deepcopy(self.wrappers)
        return new

    @property
    def parameters(self):
        return list(set.union(
            *[set(wrapper.parameters) for wrapper in self.wrappers],
            set(self.core.parameters)))

    @property
    def new(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.parameters)

    @staticmethod
    def __util_operator__(other, expr, prev, **kwargs):
        """Do `expr` action on `prev` and `other`, passing `kwargs` to `other`
        if it's callable.
        """

        return (expr(
            prev,
            other(**kwargs) if callable(other) else other))

    @staticmethod
    def __util_r_operator__(other, expr, prev, **kwargs):
        """Right operator alias for __util_r_operator__
        """

        return (expr(
            other(**kwargs) if callable(other) else other,
            prev))

    def wrap(self, fn):
        self.wrappers.append(fn)
        return self

    def __lt__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.lt))

    def __le__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.le))

    def __eq__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.eq))

    def __ne__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.ne))

    def __gt__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.gt))

    def __ge__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.ge))

    def __add__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.add))

    def __radd__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_r_operator__,
                other,
                operator.add))

    def __sub__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.sub))

    def __rsub__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_r_operator__,
                other,
                operator.sub))

    def __mul__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.mul))

    def __matmul__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.matmul))

    def __truediv__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.truediv))

    def __floordiv__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.floordiv))

    def __mod__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.mod))

    def __divmod__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                divmod))

    def __pow__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.pow))

    def __rpow__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_r_operator__,
                other,
                operator.pow))

    def __lshift__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.lshift))

    def __rlshift__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_r_operator__,
                other,
                operator.lshift))

    def __rshift__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.rshift))

    def __rrshift__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_r_operator__,
                other,
                operator.rshift))

    def __and__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.and_))

    def __xor__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.xor))

    def __or__(self, other):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                other,
                operator.or_))

    def __neg__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                operator.neg))

    def __abs__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                abs))

    def __invert__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                operator.invert))

    def __round__(self, ndigits=None):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                ndigits,
                round))

    def __trunc__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                math.trunc))

    def __floor__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                math.floor))

    def __ceil__(self):
        return self.new.wrap(
            functools.partial(
                Function.__util_operator__,
                None,
                math.ceil))
