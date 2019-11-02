"""Function classes.
"""


import inspect
import functools
import operator
import math
import physical
from copy import deepcopy


class Executable():
    """Executable code.
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

        if not all([names in kwargs for names in self.parameters]):
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


class RnFunction():
    """Multidimensional function.
    """

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
        new = RnFunction(fn=self.core.executable)
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

    def wrapWith(self, fn):
        self.wrappers.append(fn)
        return self

    def wrapWithOperator(self, other, expression, right=False):
        """Wrap function with given `expression` and `other` passed as left
        argument if right=False.
        """

        def act(other, expr, prev, **kwargs):
            return expr(
                prev,
                other(**kwargs) if callable(other) else other)

        def ract(other, expr, prev, **kwargs):
            return expr(
                other(**kwargs) if callable(other) else other,
                prev)

        return self.new.wrapWith(
            functools.partial(
                act if not right else ract, other, expression))

    def __lt__(self, other):
        return self.wrapWithOperator(other, operator.lt)

    def __le__(self, other):
        return self.wrapWithOperator(other, operator.le)

    def __eq__(self, other):
        return self.wrapWithOperator(other, operator.eq)

    def __ne__(self, other):
        return self.wrapWithOperator(other, operator.ne)

    def __gt__(self, other):
        return self.wrapWithOperator(other, operator.gt)

    def __ge__(self, other):
        return self.wrapWithOperator(other, operator.ge)

    def __add__(self, other):
        return self.wrapWithOperator(other, operator.add)

    def __radd__(self, other):
        return self.wrapWithOperator(other, operator.add, right=True)

    def __sub__(self, other):
        return self.wrapWithOperator(other, operator.sub)

    def __rsub__(self, other):
        return self.wrapWithOperator(other, operator.sub, right=True)

    def __mul__(self, other):
        return self.wrapWithOperator(other, operator.mul)

    def __matmul__(self, other):
        return self.wrapWithOperator(other, operator.matmul)

    def __truediv__(self, other):
        return self.wrapWithOperator(other, operator.truediv)

    def __floordiv__(self, other):
        return self.wrapWithOperator(other, operator.floordiv)

    def __mod__(self, other):
        return self.wrapWithOperator(other, operator.mod)

    def __divmod__(self, other):
        return self.wrapWithOperator(other, divmod)

    def __pow__(self, other):
        return self.wrapWithOperator(other, operator.pow)

    def __rpow__(self, other):
        return self.wrapWithOperator(other, operator.pow, right=True)

    def __lshift__(self, other):
        return self.wrapWithOperator(other, operator.lshift)

    def __rlshift__(self, other):
        return self.wrapWithOperator(other, operator.lshift, right=True)

    def __rshift__(self, other):
        return self.wrapWithOperator(other, operator.rshift)

    def __rrshift__(self, other):
        return self.wrapWithOperator(other, operator.rshift, right=True)

    def __and__(self, other):
        return self.wrapWithOperator(other, operator.and_)

    def __xor__(self, other):
        return self.wrapWithOperator(other, operator.xor)

    def __or__(self, other):
        return self.wrapWithOperator(other, operator.or_)

    def __neg__(self):
        return self.wrapWithOperator(None, operator.neg)

    def __abs__(self):
        return self.wrapWithOperator(None, abs)

    def __invert__(self):
        return self.wrapWithOperator(None, operator.invert)

    def __round__(self, ndigits=None):
        return self.wrapWithOperator(ndigits, round)

    def __trunc__(self):
        return self.wrapWithOperator(None, math.trunc)

    def __floor__(self):
        return self.wrapWithOperator(None, math.floor)

    def __ceil__(self):
        return self.wrapWithOperator(None, math.ceil)

    def derivative(self, variables):
        """Make partial derivative function with changes on given vars. `vars`
        is a list of names.
        """

        def d(change=1e-4, **params):
            newparams = {
                name: value + (
                    change
                    if name in variables
                    else 0)
                for (name, value)
                in params.items()
            }
            return (self(**newparams) - self(**params)) / change

        return d

    def integral(self, variables):
        """Integrate current function by given variable names.
        """

        def getByPoint(point: tuple):
            """Assign point coordinates with variable names and execute the
            function.
            """
            return self(**dict(zip(variables, point)))

        def s(space: physical.Space):
            # Check your space detalization to change step.
            result = 0
            for point in space:
                result += getByPoint(point) * \
                    (space.detalization ** len(variables))
            return result

        return s

    def antiderivative(self, variables):
        """Make antiderivative function on give vars.
        """

        fn = self.integral(variables)

        def ad(change=1e-4, **params):
            return fn(space=physical.Rectangle(
                start=(0,) * len(variables),
                # Such list comprehension guarantee that `end` point's
                # coordinates will have the same order as `variables` names.
                end=(params[name] for name in variables),
                detalization=change))

        return ad
