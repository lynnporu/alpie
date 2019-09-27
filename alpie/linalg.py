from copy import deepcopy
from math import floor, ceil
import numbers
import itertools


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

    def insertInto(self, coordinates: list, value, fillwith=None):
        """Insert given value exactly into given coordinates. Enlarge matrix by
        `fillwith` elements if it's coordinates are out of range.
        """

        def ensureIndex(array, index):
            while len(array) <= index:
                array.append(deepcopy(fillwith))
            array[index] = list()

        def insert(array, index, *tail):
            ensureIndex(array, index)
            if not tail:
                array[index] = value
            else:
                insert(array[index], *tail)

        insert(self.data, *coordinates)

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

    def mapWith(self, function, new=True):
        """Map elements of matrix with a given function and returns the result.
        If new=True, original elements won't be changed.
        """

        matrix = self if not new else deepcopy(self)

        return matrix.shape(
            fillwith=list(map(
                function, matrix.elements())),
            dimensions=matrix.dimensions,
            unpack=True)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            raise ValueError("The second variable must be number.")

        return self.mapWith(
            lambda el: el * other)

    def __add__(self, other):
        if self.dimensions != other.dimensions:
            raise ValueError("Matrices have not the same dimensions.")

        return MultidimensionalMatrix.empty().shape(
            fillwith=list(map(
                sum, zip(
                    self.elements(),
                    other.elements()))),
            dimensions=self.dimensions,
            unpack=True)

    def __sub__(self, other):
        if self.dimensions != other.dimensions:
            raise ValueError("Matrices have not the same dimensions.")

        return MultidimensionalMatrix.empty().shape(
            fillwith=list(map(
                lambda pack: pack[0] - pack[1],
                zip(
                    self.elements(),
                    other.elements()
                )
            )),
            dimensions=self.dimensions,
            unpack=True)

    def __eq__(self, other):
        return self.data == other.data

    def __neg__(self):
        return self * -1

    def __abs__(self):
        return self.mapWith(abs)

    def __round__(self, n):
        return self.mapWith(
            lambda el: round(el, n))

    def __floor__(self):
        return self.mapWith(floor)

    def __ceil__(self):
        return self.mapWith(ceil)

    def __int__(self):
        return self.mapWith(int)

    def __float__(self):
        return self.mapWith(float)

    def __complex__(self):
        return self.mapWith(complex)

    def __oct__(self):
        return self.mapWith(oct)

    def __hex__(self):
        return self.mapWith(hex)

    def __str__(self):
        out = str()
        for row in self.data:
            out += str(row) + "\n"
        return out

    def __repr__(self):
        return f"<MultidimensionalMatrix dimensions={self.dimensions}>"

    def __len__(self):
        """Returns size of dimensions.
        """
        return len(self.dimensions)

    def __deepcopy__(self, memdict):
        return type(self)(data=deepcopy(self.data))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __hash__(self):
        return int(
            "".join(
                map(
                    lambda n: f'{n:03}',
                    itertools.chain(self.dimensions, self.elements()))))

    def wrapWith(self, cls):
        """Create new exemplar of given class with data of current one.
        """
        return cls(data=self.data)

    @property
    def to2d(self):
        """Returns Matrix with the same data.
        """
        if len(self.dimensions) != 2:
            raise ValueError("This matrix is not 2-dimensional")

        return Matrix(data=self.data)


class Matrix(MultidimensionalMatrix):

    def __init__(self, height=None, width=None, data=None):
        """Create [height x width] matrix.
        """
        if type(data) is list:
            self.data = data
        else:
            return super().__init__([height, width], data)

    def rows(self):
        """Generate rows.
        """
        for row in self.data:
            yield row

    def columns(self):
        """Generate columns.
        """
        for n in range(self.dimensions[1]):
            yield [row[n] for row in self.data]

    def getRow(self, n):
        """Returns n-th row.
        """
        return self.data[n]

    def getColumn(self, n):
        """Returns n-th column.
        """
        return [row[n] for row in self.data]

    @property
    def isSquare(self):
        dimensions = self.dimensions
        # Check if list has two elements and they are same.
        return dimensions.count(dimensions[0]) == len(dimensions) == 2

    def toSquare(self):
        """Return SquareMatrix with the data of a current one.
        """
        if not self.isSquare:
            raise TypeError("This matrix is not square.")

        return SquareMatrix(data=self.data)

    @property
    def transposed(self):
        return type(self)(data=[list(row) for row in zip(*self.data)])

    def insertInto(value, coordinates, fillwith):
        if len(coordinates) > 2:
            raise ValueError("Such dimension can not exist in this matrix.")
        else:
            super().insertInto(value, coordinates, fillwith)

    @property
    def euclideanNorm(self):
        return sum(
            map(
                lambda n: n ** 2,
                self.elements())) ** .5

    def __mul__(self, other):
        # Try scalar multiplication.
        try:
            return (MultidimensionalMatrix(data=self.data) * other).to2d

        # If not:
        except ValueError:

            if self.dimensions[1] != other.dimensions[0]:
                raise ValueError("These matrices can not be multiplied.")

            return Matrix(
                data=[
                    [
                        sum(
                            el1 * el2
                            for el1, el2
                            in zip(row1, col2)
                        )
                        for col2
                        in zip(*other.data)
                    ]
                    for row1
                    in self.data
                ]
            )

    def __pow__(self, n):
        if n == -1:
            return self.inverse()
        elif n < -1:
            raise ValueError("Operation is not determined.")

        new = deepcopy(self)

        while n > 1:
            n -= 1
            new *= new

        return new

