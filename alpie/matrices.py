"""This module includes classes of matrices.
"""


from copy import deepcopy
import pprint
import math
import numbers
import itertools
import operator


class InappropriateDimensions(Exception):
    pass


class Matrix:
    """This class represents a matrix of any type and shape.

    Methods should be implemented for this interface:
        @classmethod ensure(cls, data, *args, **kwargs)
            Ensure if given `data` object has the same class as the current. If
            yes, then return copy of it or try to create
            cls(data, *args, **kwargs) otherwise.
        @property new:
            Returns object of current class.
        @classmethod empty(cls):
            Returns empty object of this class.
        @property elements:
            Generates elements of the matrix in the recursive order
        __eq__(self, other)
        __bool__(self)
        __str__(self)
        __repr__(self):
        __deepcopy__(self, memdict)
        __getitem__(self, key)
        __setitem__(self, key, item)

    """

    @classmethod
    def ensure(cls, data, *args, **kwargs):

        if isinstance(data, cls):
            return deepcopy(data, *args, **kwargs)

        else:
            return cls(data)

    @property
    def new(self):
        return type(self)

    @classmethod
    def empty(cls, *args, **kwargs):
        """Creates empty matrix.
        """
        return cls(*args, **kwargs)

    @property
    def elements(self):
        raise NotImplemented

    def __eq__(self, other):
        raise NotImplemented

    def __bool__(self):
        raise NotImplemented

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        return f"<Matrix>"

    def __deepcopy__(self, memdict):
        raise NotImplemented

    def __getitem__(self, key):
        raise NotImplemented

    def __setitem__(self, key, item):
        raise NotImplemented


class ListMatrix(Matrix):
    """This interface defines a matrix which elements should be stored in a
    single list object which is usually not one-dimensional. For example, all
    the elements of the diagonal or tridiagonal matrix you can store with a
    bunch (or just a single) sequence of numbers.

    Properties:
        list data:
            Usually the place where elements stored in.
        tuple dimensions:
            Tuple of dimensions of this matrix.

    """

    def __init__(self, data: list):
        """Create a matrix with given data.
        """
        self.data = deepcopy(data)

    @classmethod
    def ensure(cls, data):

        if isinstance(data, cls):
            return data
        else:
            return cls(data)

    @classmethod
    def sizedAs(cls, dimensions: tuple):
        """Creates matrix filled with None objects with given dimension.
        """
        return cls.shape(None, dimensions)

    def clear(self):
        """Clear self.data.
        """
        self.data = list()

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

        self.clear()

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

            # Go recursively.
            for item in array:
                measure(item, depth + 1)

        measure(self.data)

        return tuple(coords)

    @classmethod
    def empty(cls):
        return cls.new(list())

    @property
    def sketch(self):
        """Returns empty matrix with same dimensions.
        """
        return self.new.sizedAs(self.dimensions)

    @property
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

    @property
    def coordinates(self):
        """Return iterator of all coordinates in matrix.
        """
        return itertools.product(
            *map(range, self.dimensions))

    def at(self, *coordinates):
        """Return element at given coordinates. Useful for indexing with tuples:
        A[0][1][2] -> A.at(0, 1, 2)
        """

        def dig(array, index, *tail):
            if not tail:
                return array[index]
            else:
                return dig(array[index], *tail)

        return dig(self.data, *coordinates)

    @property
    def enum(self):
        """Return iterator of all elements with its coordinates in matrix.
        """
        return [
            (self.at(*position), position)
            for position
            in self.coordinates]

    def mapWith(self, function, new=True):
        """Map elements of matrix with a given function and returns the result.
        If new=True, original elements won't be changed.
        """

        matrix = self if not new else deepcopy(self)

        return matrix.shape(
            fillwith=list(map(
                function, matrix.elements())),
            unpack=True)

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

        return self.new(new)

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

    def __eq__(self, other):
        if not isinstance(other, ListMatrix):
            raise TypeError("Can't compare ListMatrix with something else.")

        return self.data == other.data

    def __bool__(self):
        return bool(self.data)

    def __str__(self):
        return pprint.pprint(self.data)

    def __repr__(self):
        return f"<ListMatrix dimensions={self.dimensions}>"

    def __deepcopy__(self, memdict):
        return self.new(deepcopy(self.data))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __mul__(self, other):
        # Scalar multiplication of two matrices.
        if (
            not isinstance(other, numbers.Number) and
            self.dimensions == other.dimensions
        ):
            return sum(map(
                operator.mul, self.elements(), other.elements()))

        # Multiply by number.
        else:
            return self.mapWith(
                lambda el: el * other)

    def __truediv__(self, other):
        return self.mapWith(
            lambda el: el / other)

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
            fillwith=list(map(
                operator.sub,
                self.elements(), other.elements())),
            unpack=True)

    def __neg__(self):
        return self * -1

    def __abs__(self):
        return self.mapWith(abs)

    def __round__(self, n):
        return self.mapWith(
            lambda el: round(el, n))

    def __floor__(self):
        return self.mapWith(math.floor)

    def __ceil__(self):
        return self.mapWith(math.ceil)

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

    def __len__(self):
        """Returns first dimension.
        """
        return len(self.data)

    def convertTo(self, cls):
        """Create new exemplar of given class with data of current one.
        """
        return cls(self.data)


class Matrix(MultidimensionalMatrix):

    def __init__(self, height=None, width=None, data=None):
        """Create [height x width] matrix.
        """
        if isinstance(data, list):
            self.data = data
        else:
            return super().__init__([height, width], data)

    def __repr__(self):
        return f"<Matrix dimensions={self.dimensions}>"

    @classmethod
    def sizedAs(cls, dimensions: tuple):
        """`dimensions` is (height, width)
        """
        height, width = dimensions
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
            return self.new.filledWith(
                [[el] for el in self.data]
            )

        else:
            return self.new.filledWith(
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

    def __matmul__(self, other):
        if self.dimensions[1] != other.dimensions[0]:
            raise ValueError("These matrices can not be multiplied.")

        # Result might have another dimension than multipliers
        return Matrix.filledWith(
            [
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


class SquareMatrix(Matrix):

    def __init__(self, size=None, data=None):
        """Create [size x size] matrix.
        """
        if isinstance(size, list):
            size = size[0]
        if isinstance(data, list) and not size:
            self.data = data
            return
        else:
            return super().__init__(size, size, data)

    def __repr__(self):
        return f"<SquareMatrix size={len(self)}>"

    @classmethod
    def filledWith(cls, data):
        return cls(size=None, data=data)

    @classmethod
    def sizedAs(cls, *size):
        if type(size) is list:
            size = size[0]
        return cls(size=size, data=None)

    @classmethod
    def givens(cls, size, i, j, theta):
        """Returns Givens matrix of given size.
        """
        matrix = cls.ofIdentity(size=size)
        matrix[i][i] = matrix[j][j] = math.cos(theta)
        matrix[i][j] = -math.sin(theta)
        matrix[j][i] = math.sin(theta)
        return matrix

    def givensMask(self, i, j, theta):
        """Returns Givens matrix with size of current matrix.
        """
        return self.new.givens(len(self), i, j, theta)

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

    def zeroDiagonal(self):
        """Make diagonal of this matrix equal to zero.
        """
        for i in range(len(self)):
            self[i][i] = 0


class AugmentedMatrix:

    # TODO: add `ensure` method.
    def __init__(self, coeffs, eqs, forwardRootsOrder=True):
        """Check A and B matrices and create augmented matrix.

        `forwardRootsOrder` == False means that calculated x1..xn should be
        inverted into xn..x1 in order to represent real X vector.
        This can happen when columns of coefficients matrix was swapped.
        """
        coeffs = Matrix.ensure(coeffs)
        eqs = Matrix.ensure(eqs)

        if eqs.dimensions != [coeffs.dimensions[0], 1]:
            raise ValueError("Incorrect system.")

        self.coeffs = coeffs
        self.eqs = eqs
        self.forwardRootsOrder = forwardRootsOrder

    @property
    def new(self):
        return type(self)

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
            raise ValueError("Must be AugmentedMatrixRow.")
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
        return self.new(
            **deepcopy(self.__dict__))

    def calculate(self, x):
        """Calculate result of equations with given X vector.
        """

        return self.coeffs @ x

    def fixedPointIteration(self, initial, iterator, accuracyfunc):
        """Solve system of linear equations by fixed-point iteration method in
        form of: x(k+1) = x(k) * C + d

        accuracyfunc will be called with (x0, x1) parameters, where each of
        them define X vector at correspond step.

        iterator will be called with X-vector at previous step. It should
        return new X.
        """
        X = deepcopy(initial)

        while True:
            newX = iterator(X)
            if accuracyfunc(X, newX):
                X = newX
                break
            else:
                X = newX

        return X


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


class NotSolvable(Exception):
    pass


# TODO: Test this class.
class EliminatedAugmentedMatrix(AugmentedMatrix):

    def __init__(self, coeffs, eqs, forwardRootsOrder=True):
        coeffs = TriangularMatrix.ensure(coeffs)

        return super().__init__(coeffs, eqs, forwardRootsOrder)

    def __repr__(self):
        return (
            f"<EliminatedAugmentedMatrix equations={len(self.coeffs)} "
            f"variables={self.coeffs.dimensions[1]}>")

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
        """Calculate roots out of upper-right matrix. Check TriangularMatrix
        docstrings for definition.
        """

        matrix = self.upperRight

        x = list()

        for row, eq in zip(reversed(self.coeffs), reversed(self.eqs)):

            seq = 0

            # Multiplying found roots vector with row of coefficients and sum
            # it up.
            for index, coeff in enumerate(reversed(row)):

                if len(x) > index:
                    seq += coeff * x[index]

                else:
                    x.append((eq[0] - seq) / coeff)
                    break

        return Matrix.filledWith(
            x[::-1] if matrix.forwardRootsOrder else x).transposed


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

    @property
    def det(self):
        return self.sign * self.diagmul

    def __deepcopy__(self, memdict):
        return self.new(**deepcopy(self.__dict__))

    def __setitem__(self, *args):
        raise TypeError("Assigning is not allowed.")
