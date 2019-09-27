from copy import deepcopy
class MultidimensionalMatrix:

    def __init__(self, dimensions=None, data=None):
        """Create matrix with given dimensions and fill it with None OR create
        matrix and assign it with data.
        """
        if not dimensions and type(data) is list:
            self.data = deepcopy(data)

        elif not data:
            self.shape(None, dimensions)

        else:
            raise ValueError("Check docstring for help.")

    @classmethod
    def empty(cls):
        """Creates empty matrix.
        """
        return cls(data=[])

    def elements(self):
        """Generates elements of matrix in the recursive order.
        """
        def flat(array):
            for el in array:
                if type(el) is list:
                    for subel in flat(el):
                        yield subel
                else:
                    yield el

        return flat(self.data)

