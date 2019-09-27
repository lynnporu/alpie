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

    def shape(
        self, fillwith=None, dimensions=None, unpack=False,
        background=None
    ):
        """If unpack==False, then matrix will be filled with copies of
        `fillwith` object.
        If unpack==True, then elements of `fillwith` will be shaped into matrix
        with given dimensions in recursive order. Missing elements will be
        filled as `background`, otherwise ValueError will raise.
        """
        self.data = list()

        def fill(array, size, *coords):

            for _ in range(size):

                # We're at the lowest level.
                if not coords:
                    if not unpack:
                        array.append(deepcopy(fillwith))
                    else:
                        if not fillwith:
                            array.append(deepcopy(background))
                        else:
                            array.append(fillwith.pop(0))

                # Unpack coords list to size + coords
                else:
                    item = list()
                    fill(item, *coords)
                    array.append(item)

        fill(self.data, *dimensions)

        return self

    @classmethod
    def empty(cls):
        """Creates empty matrix.
        """
        return cls(data=[])

    @property
    def dimensions(self):
        """Measure dimensions of data.
        """
        coords = list()

        def measure(array, depth=0):

            if type(array) is not list:
                return

            # Check if coords list are big enough.
            if len(coords) - 1 < depth:
                coords.append(0)

            # Add size of the biggest element at this dimension.
            coords[depth] = max(coords[depth], len(array))

            # Go recursively.
            for item in array:
                measure(item, depth + 1)

        measure(self.data)

        return coords

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

