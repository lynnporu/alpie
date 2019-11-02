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

    def __len__(self):
        """Returns function dimensionality.
        """
        return len(self.parameters)


class RnFunction():
    """Multidimensional function.
    """

    def __init__(self, fn):
        """Creates function from given callable object.
        """
        self.core = Executable(fn)
        self.wrappers = []

    @classmethod
    def ensure(cls, data):
        """Ensure if `obj` has type of this class.
        """
        if isinstance(data, cls):
            return data

        else:
            return cls(fn=data)

    def __repr__(self):
        return "<{} core with{}wrappers>".format(
            self.core.__repr__(),
            " no " if not self.wrappers else " ")

    def __hash__(self):
        return hash(self.core, self.wrappers)

    def __call__(self, **kwargs):
        result = self.core(**kwargs)
        return self.calculateWrappers(result, **kwargs)

    def calculateWrappers(self, initial, **kwargs):
        """Calculate initial value by all the wrappers of this function.
        """

        for fn in self.wrappers:
            initial = fn(initial, **kwargs)

        return initial

    def __deepcopy__(self, memdict):
        new = type(self)(fn=self.core.executable)
        new.wrappers = deepcopy(self.wrappers)
        return new

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

    def derivative(self, variables, change=1e-4):
        """Make partial derivative function with changes on given vars. `vars`
        is a list of names.

        Result is self instance.
        """

        def differentialWrapper(fn, change, variables, prev, **params):
            newparams = {
                # Add change only of name in variables.
                # a * False = 0, a * True = a
                name: value + change * (name in variables)
                for name, value
                in params.items()
            }
            return (fn(**newparams) - prev) / change

        return self.new.wrapWith(
            functools.partial(
                differentialWrapper, self, change, variables))

    def integral(self, variables):
        """Integrate current function by given variable names.
        """

        def getByPoint(point: tuple):
            """Assign point coordinates with variable names and execute the
            function.
            """
            if isinstance(point, tuple):
                return self(
                    **dict(zip(variables, point)))
            else:
                return self(
                    **{name: point for name in variables})

        def worker(space: physical.Space):
            # Check your space detalization to change step.
            result = 0
            for point in space:
                result += getByPoint(point) * \
                    (space.detalization ** len(variables))
            return result

        return worker

    def antiderivative(self, variables, change=1e-4):
        """Make antiderivative function on give vars.
        """

        def integralWrapper(integral, change, variables, prev, **params):
            return integral(space=physical.Rectangle(
                start=(0,) * len(variables),
                # Such list comprehension guarantee that `end` point's
                # coordinates will have the same order as `variables` names.
                end=(params[name] for name in variables),
                detalization=change))

        return self.new.wrapWith(
            functools.partial(
                integralWrapper, self.integral(variables), change, variables))


class Function(RnFunction):
    """One-dimensional function f(x) = y.
    """

    def __init__(self, fn):

        self.core = Executable(fn)
        if len(self.core) > 1:
            raise ValueError(
                "You should provide one-dimensional function.")

        self.wrappers = []
        self.parameter = self.core.parameters[0]

    def wrapWithOperator(self, other, expression, right=False):
        """Wrap function with given non-callable expression.
        """

        if callable(other):
            raise ValueError(
                "You should create multidimensional function in order to use"
                " in functional expressions.")

        super().wrapWithOperator(self, other, expression, right)

    def __call__(self, value=None, **kwargs):
        params = kwargs if value is None else {self.parameter: value}
        result = self.core(**params)
        return self.calculateWrappers(result, **params)

    def derivative(self, change=1e-4):
        return super().derivative([self.parameter], change)

    def integral(self):
        return super().integral([self.parameter])

    def antiderivative(self, change=1e-4):
        return super().antiderivative([self.parameter], change)

    def findByComparison(self, space: physical.Range, expr):
        """Iterates over given space with given algorithm:

        found = 0
        for value in space:
            found = expr(found, self(value))
        return found

        Can be used in order to find maximum or minimum:
        f.findByComparison(Range(0, 10), max)

        """

        if not isinstance(space, physical.Range):
            raise TypeError("Give me physical.Range object.")

        if not callable(expr):
            raise TypeError("expr must be callable.")

        found = None
        for value in space:
            # Pass first value
            if found is None:
                found = self(value)
                continue
            found = expr(found, self(value))

        return found
