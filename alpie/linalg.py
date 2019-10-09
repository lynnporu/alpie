from copy import deepcopy
from math import floor, ceil
import numbers
import itertools
import operator


class InappropriateDimensions(Exception):
    pass


class MultidimensionalMatrix:

    def __init__(self, dimensions=None, data=None):
        """Create matrix with given dimensions and fill it with None OR create
        matrix and assign it with data.
        """
        if not dimensions and isinstance(data, list):
            self.data = deepcopy(data)

        elif not dimensions and not data:
            raise ValueError("Check docstring for help.")

        else:
            self.shape(data, dimensions)

    @classmethod
    def sizedAs(cls, dimensions):
        return cls(dimensions, None)

    @classmethod
    def filledWith(cls, data):
        return cls(None, data)

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
        if not dimensions:
            dimensions = self.dimensions

        self.data = list()

        def fill(array, size, *coords):

            for _ in range(size):

                # We're at the lowest level.
                if not coords:
                    if not unpack:
                        array.append(deepcopy(fillwith))
                    else:
                        array.append(
                            deepcopy(background)
                            if not fillwith else
                            fillwith.pop(0)
                        )

                # Unpack coords list to size + coords
                else:
                    item = list()
                    fill(item, *coords)
                    array.append(item)

        fill(self.data, *dimensions)

        return self

    @property
    def sketch(self):
        """Returns empty matrix with same dimensions.
        """
        return type(self).sizedAs(self.dimensions)

    @classmethod
    def empty(cls):
        """Creates empty matrix.
        """
        return cls.filledWith([])

    @property
    def dimensions(self):
        """Measure dimensions of data.
        """
        coords = list()

        def measure(array, depth=0):

            if not isinstance(array, list):
                return

            # Check if `coords` list are big enough.
            if len(coords) - 1 < depth:
                coords.append(0)

            # Add size of the biggest element at this dimension.
            coords[depth] = max(coords[depth], len(array))

            # TODO: check if array is list
            # Go recursively.
            for item in array:
                measure(item, depth + 1)

        measure(self.data)

        return coords

    def inversedAt(self, *dimensions):
        """Inverse matrix at each given dimension depth.
        Example:
            inverseAt(0) will inverse rows of square matrix,
            inverseAt(1) will inverse columns of square matrix,
            inverseAt(0, 1) returns transposed matrix.

        If dimensions is list, then matrix will be inversed on each of
        dimensions consequently.
        """

        def dig(array, lookfor, currdepth=0):

            if currdepth == lookfor:
                return array[::-1]

            else:
                array = [
                    dig(subarr, lookfor, currdepth + 1)
                    for subarr in array
                ]

            return array

        new = deepcopy(self.data)

        try:

            for dim in dimensions:
                new = dig(new, dim)

        # Object is not iterable.
        except TypeError:
            raise InappropriateDimensions

        return type(self).filledWith(new)

    def inverseAt(self, *dimensions):
        """Alias for `inversedAt` method.
        """
        self.data = self.inversedAt(*dimensions).data

    @property
    def rotated(self):
        """Inverse this matrix at each dimension level.
        """
        return self.inversedAt(*self.dimensions)

    def rotate(self):
        """Make this matrix rotated.
        """
        self = self.rotated

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
                if isinstance(el, list):
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
            unpack=True)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            raise InappropriateDimensions

        return self.mapWith(
            lambda el: el * other)

    def __add__(self, other):
        if self.dimensions != other.dimensions:
            raise InappropriateDimensions

        return self.sketch.shape(
            fillwith=list(map(
                operator.add,
                self.elements(), other.elements())),
            unpack=True)

    def __sub__(self, other):
        if self.dimensions != other.dimensions:
            raise InappropriateDimensions

        return self.sketch.shape(
            fill=list(map(
                operator.sub,
                self.elements(), other.elements())),
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
        return self.dimensions[0]

    def __deepcopy__(self, memdict):
        return type(self).filledWith(deepcopy(self.data))

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

    def convertTo(self, cls):
        """Create new exemplar of given class with data of current one.
        """
        return cls.filledWith(self.data)


class Matrix(MultidimensionalMatrix):

    def __init__(self, height=None, width=None, data=None):
        """Create [height x width] matrix.
        """
        if isinstance(data, list):
            self.data = data
        else:
            return super().__init__([height, width], data)

    @classmethod
    def sizedAs(cls, height, width):
        return cls(height=height, width=width, data=None)

    @classmethod
    def filledWith(cls, data):
        return cls(height=None, width=None, data=data)

    @classmethod
    def zeros(cls, height, width):
        return cls(height, width, data=0)

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
    def transposed(self):
        if len(self.dimensions) == 1:
            return type(self).filledWith(
                [[el] for el in self.data]
            )

        else:
            return type(self).filledWith(
                [list(row)
                    for row in
                    zip(*self.data)])

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
            return (
                MultidimensionalMatrix.filledWith(self.data) * other
            ).convertTo(Matrix)

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
            return self.inversed
        elif n < -1:
            raise ValueError("Operation is not determined.")

        new = deepcopy(self)

        while n > 1:
            n -= 1
            new *= new

        return new


class NotSymmetric(Exception):
    pass


class SquareMatrix(Matrix):

    def __init__(self, size=None, data=None):
        """Create [size x size] matrix.
        """
        if isinstance(data, list) and not size:
            self.data = data
            return
        else:
            return super().__init__(size, size, data)

    @classmethod
    def filledWith(cls, data):
        return cls(size=None, data=data)

    @classmethod
    def sizedAs(cls, size):
        return cls(size=size, data=None)

    @classmethod
    def zeros(cls, size):
        return cls(size=size, data=0)

    @property
    def diagonal(self):
        """Generate diagonal of the matrix.
        """
        return (
            row[i]
            for i, row
            in enumerate(self.data))

    @property
    def antidiagonal(self):
        """Generate antidiagonal of the matrix.
        """
        return (
            row[len(self.data) - i]
            for i, row
            in enumerate(self.data))

    @property
    def diagmul(self):
        """Multiply elements of diagonal.
        """
        result = 1
        for el in self.diagonal:
            result += el
        return result

    @property
    def inversed(self, *args):
        new = SquareMatrix.empty()

        for vector in self.identityMask.rows():
            new.data.append(
                AugmentedMatrix(
                    coeffs=self,
                    eqs=Matrix.filledWith(vector).transposed)
                .gaussianEliminated()
                .roots)

        return new

    @property
    def cond(self):
        return self.euclideanNorm * self.inversed.euclideanNorm

    @classmethod
    def ofIdentity(cls, size):
        return cls.filledWith([
            [
                1 if ncell == nrow else 0
                for ncell
                in range(size)
            ]
            for nrow
            in range(size)
        ])

    @property
    def choleskyDecomposed(self):
        if not self.isSymmetric:
            raise NotSymmetric("This method require symmetrical matrix.")

        n = self.dimensions[0]

        T = SquareMatrix(size=n, data=0)

        T[0][0] = self[0][0] ** .5

        for i, j in itertools.product(range(0, n), repeat=2):

            if i == 0:

                T[i][j] = self[i][j] / T[i][i]

            elif i == j:

                S = 0
                for k in range(0, i):
                    S += T[k][i] ** 2
                T[i][j] = (self[i][j] - S) ** .5

            elif i < j:

                S = 0
                for k in range(0, i):
                    S += T[k][i] ** 2
                T[i][j] = (self[i][j] - S) / T[i][i]

            elif i > j:

                T[i][j] = 0

        return (T.transposed, T)

    @property
    def identityMask(self):
        return SquareMatrix.ofIdentity(size=len(self))

    @property
    def isSymmetric(self):
        # It could be shorter with self == self.transposed, but that means
        # instantiating another matrix class and comparing them, so the
        # following approach is faster.
        square = list(map(tuple, self.data))
        return square == list(zip(*square))

    @property
    def withInversedRows(self):
        return self.inversedAt(0)

    def inverseRows(self):
        self.inverseAt(0)

    @property
    def withInversedColummns(self):
        return self.inverseAt(1)

    def inverseColumns(self):
        self.inverseAt(1)


class NotSolvable(Exception):
    pass


class AugmentedMatrix():

    def __init__(self, coeffs, eqs, forwardRootsOrder=True):
        """Check A and B matrices and create augmented matrix.

        `forwardRootsOrder` == False means that calculated x1..xn should be
        inverted into xn..x1 in order to represent real X vector.
        This can happen when columns of coefficients matrix was swapped.
        """
        if not(
            isinstance(coeffs, TriangularMatrix) and isinstance(eqs, Matrix)
        ):
            raise ValueError(
                "Coefficients must be TriangularMatrix, B vector is Matrix.")

        if eqs.dimensions != [coeffs.dimensions[0], 1]:
            raise ValueError("Incorrect system.")

        self.coeffs = coeffs
        self.eqs = eqs
        self.forwardRootsOrder = forwardRootsOrder

    def inverseColumns(self):
        """Inverse columns of coefficients matrix.
        """
        self.coeffs.inverseColumns()
        self.forwardRootsOrder = not self.forwardRootsOrder

    def inverseRows(self):
        """Inverse columns of coefficients and `eqs` matrix.
        """
        self.coeffs.inverseRows()
        self.eqs.inverseAt(0)

    @classmethod
    def withZeroEqs(cls, coeffs):
        return cls(
            coeffs,
            eqs=Matrix(
                data=[[1]] * coeffs.dimensions[0]))

    def __repr__(self):
        return (
            f"<AugmentedMatrix equations={len(self.coeffs)} "
            f"variables={self.coeffs.dimensions[1]}>"
        )

    def __getitem__(self, key):
        return AugmentedMatrixRow(self.coeffs[key], self.eqs[key])

    def __setitem__(self, key, item):
        if not isinstance(item, AugmentedMatrixRow):
            raise ValueError("Assigning item type must be AugmentedMatrixRow.")

        self.coeffs[key], self.eqs[key] = item.coeffs, item.eq

    def __len__(self):
        return self.coeffs.dimensions[0]

    def __str__(self):
        out = str()
        for coeffs, equation in zip(self.coeffs, self.eqs):
            out += "\t".join(map(str, coeffs)) + \
                "\t|\t" + str(equation[0]) + "\n"
        return out

    def __deepcopy__(self, memdict):
        return type(self)(
            **deepcopy(self.__dict__))

    def gaussianEliminated(self, rowSorting=True):
        """Gaussian elimination.
        """

        new = deepcopy(self)
        sign = 1

        n = len(new)

        for i in range(0, n - 1):
            # Diagonal element.
            x = abs(new[i][i])
            maxRow = i

            if rowSorting:
                for k in range(i + 1, n):
                    if abs(new[k][i]) > x:
                        x = abs(new[k][i])
                        maxRow = k

            if i != maxRow:
                new[i], new[maxRow] = new[maxRow], new[i]
                sign *= -1

            if abs(new[i][i]) < 1e-12:
                break

            # Gaussian elimination
            for k in range(i + 1, n):
                c = -new[k][i] / new[i][i]
                for j in range(i, n + 1):
                    new[k][j] += c * new[i][j]

        new.coeffs = TriangularMatrix(data=new.coeffs.data, sign=sign)

        return new

    def gaussianEliminate(self, *args, **kwargs):

        self = self.gaussianEliminated(*args, **kwargs)

    @property
    def upperRight(self):
        """Return copy of this matrix with upper-right coefficients. This can
        be used in order to calculate roots.
        """
        new = deepcopy(self)

        if new.coeffs.isUpperRight:
            return new

        if new.coeffs.isUpperLeft:
            new.inverseColumns()
        elif new.coeffs.isLowerLeft:
            new.inverseColumns()
            new.inverseRows()
        elif new.coeffs.isLowerRight:
            new.inverseRows()

        return new

    @property
    def roots(self):
        """Calculate roots out of upper-right matrix. Look TriangularMatrix
        docstrings for definition.
        """

        matrix = self.upperRight

        n = len(matrix)
        x = []

        for i in range(n - 1, -1, -1):

            try:
                x.insert(0, matrix[i][n] / matrix[i][i])
            except ZeroDivisionError:
                raise NotSolvable

            for k in range(i - 1, -1, -1):
                matrix[k][n] -= matrix[k][i] * x[0]

        return x if matrix.forwardRootsOrder else x[::-1]


class AugmentedMatrixRow:

    def __init__(self, coeffs, eq):

        self.coeffs = coeffs
        self.eq = eq

    def __getitem__(self, key):

        if key <= len(self.coeffs) - 1:
            return self.coeffs[key]

        else:
            return self.eq[0]

    def __setitem__(self, key, item):

        if key <= len(self.coeffs) - 1:
            self.coeffs[key] = item

        else:
            self.eq[0] = item

    def __str__(self):
        return str(self.coeffs + self.eq)

    def __repr__(self):
        return f"<AugmentedMatrixRow data={str(self)}>"


class TriangularMatrix(SquareMatrix):

    def __init__(self, data, size=None, sign=1):
        if size:
            raise TypeError("Size doesn't matter.")

        self.sign = sign
        self.data = data

        if self.lookCorners == [True] * 4:
            raise ValueError("Given data is not triangular.")

    def __repr__(self):
        return f"<TriangularMatrix dimensions={self.dimensions}>"

    @property
    def lookCorners(self):
        """Return list of 4 elements, where each one represents whether the
        value of a specific corner is presented. The result has such format:
        [upper-left, upper-right, lower-left, lower-right].
        For example, if we have such matrix:
        1 1 ... 1
        0 1 ... 1
        ...
        0 0 ... 0
        then function will return [True, True, False, True]
        """
        return [
            self[0][0] != 0,
            self[0][-1] != 0,
            self[-1][0] != 0,
            self[-1][-1] != 0
        ]

    @property
    def isUpperLeft(self):
        return self.lookCorners == [True, True, True, False]

    @property
    def isUpperRight(self):
        return self.lookCorners == [True, True, False, True]

    @property
    def isLowerLeft(self):
        return self.lookCorners == [True, False, True, True]

    @property
    def isLowerRight(self):
        return self.lookCorners == [False, True, True, True]

    @classmethod
    def byGauss(cls, matrix):
        return AugmentedMatrix.withZeroEqs(matrix).gaussianEliminated().coeffs

    @property
    def det(self):
        return self.sign * self.diagmul

    def __deepcopy__(self, memdict):
        return type(self)(**deepcopy(self.__dict__))

    def __setitem__(self, *args):
        raise TypeError("Assigning is not allowed.")
