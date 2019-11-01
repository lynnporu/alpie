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
