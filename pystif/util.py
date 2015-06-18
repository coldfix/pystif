from functools import partial
from os import path
import sys
import numpy as np
from .core.util import format_vector, make_int_exact, scale_to_int
from .core.lp import Problem


def detect_prefix(s, prefix, on_prefix):
    """
    Check whether ``s`` starts with ``prefix``. If so, call ``on_prefix`` with
    the stripped string.
    """
    if s.startswith(prefix):
        if on_prefix:
            on_prefix(s[len(prefix):])
        return True
    return False


def remove_comments(lines, on_comment=None):
    """Iterate over non-comment lines. Forward comments to ``on_comment``."""
    for line in lines:
        if not detect_prefix(line.strip(), '#', on_comment):
            yield line


def read_system(filename, *, ndmin=2):
    """Read linear system from file, return tuple (colnames, matrix)."""
    comments = []
    with open(filename) as f:
        contents = remove_comments(f, comments.append)
        matrix = np.loadtxt(contents, ndmin=ndmin)
    cols = []
    for line in comments:
        detect_prefix(line.strip(), '::', lambda s: cols.extend(s.split()))
    return matrix, cols or None


def _unique(items):
    ret = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ret.append(item)
    return ret


class System:

    """
    IO utility for systems. Keeps track of column names.
    """

    def __init__(self, matrix=None, columns=None, io_order=None):
        self.matrix = matrix
        self._columns = None
        self._slice = None
        if columns:
            self.columns = columns
        if io_order:
            self.io_order = io_order
        self._seen = VectorMemory()
        self._file = None
        if matrix is not None:
            self._seen.add(*matrix)

    @classmethod
    def load(cls, filename=None, *, default=sys.stdin, force=True):
        if filename == '-' or not filename:
            return cls()
        if not force and not path.exists(filename):
            return cls()
        matrix, io_order = read_system(filename)
        return cls(matrix, io_order=io_order)

    @classmethod
    def save(cls, filename=None, *, default=sys.stdout, append=False):
        if append:
            system = cls.load(filename, force=False)
        else:
            system = cls()
        if filename and filename != '-':
            system._file = open(filename, 'a' if append else 'w')
        else:
            system._file = default
        return system

    @property
    def io_order(self):
        if self._slice:
            return [self._columns[i] for i in self._slice]
        return self._columns

    @io_order.setter
    def io_order(self, columns):
        """Set the column order for physical I/O."""
        assert self._slice is None, \
            "Can't change I/O order in append mode or after writing a row!"
        self._update_io_order(columns)

    def _update_io_order(self, columns):
        if not self._columns:
            self._columns = columns
        self._slice = [self._get_column_index(c) for c in columns]

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        """Define the column names of newly added vectors."""
        columns = list(columns)
        if self._columns is None:
            self._columns = columns
            return
        assert set(columns) == set(self._columns)
        if self._matrix is not None:
            translate = [self._columns.index(c) for c in columns]
            self.matrix = self.matrix[:,translate]
        if self._slice:
            io_order = self.io_order
            self._columns = columns
            self._update_io_order(io_order)
        else:
            self._columns = columns

    @property
    def dim(self):
        if self.matrix is not None:
            return self.matrix.shape[1]
        else:
            return len(self.columns)

    def slice(self, columns, fill=False):
        """Return reordered system. ``fill=True`` appends missing columns."""
        indices = [0] + [self._get_column_index(c) for c in columns]
        indices = _unique(indices)
        subdim = len(indices)
        if fill:
            indices += sorted(set(range(self.dim)) - set(indices))
        if self.columns:
            columns = [self.columns[i] for i in indices]
        else:
            columns = None
        sys = System(self.matrix[:,indices], columns)
        sys.subdim = subdim
        return sys

    def _get_column_index(self, col):
        try:
            return int(col)
        except ValueError:
            return self.columns.index(col)

    def add(self, v):
        """Output the vector ``v``."""
        if self._seen(v):
            return
        if self.matrix is None:
            self.matrix = np.array([v])
        else:
            self.matrix = np.vstack((self.matrix, v))
        if self._slice:
            v = v[self._slice]
        self._print(v)

    def lp(self):
        """Get the LP."""
        return Problem(self.matrix)

    def _print(self, v):
        if not self._file:
            return
        if self.matrix.shape[0] == 1 and self.io_order:
            print('#::', *self.io_order, file=self._file)
        print(format_vector(v), file=self._file)


def print_to(filename=None, *default_prefix,
             append=False, default=sys.stdout):
    """
    Return a print function that prints to filename.

    If filename is left empty, prints to STDOUT.
    """
    if filename and filename != '-':
        mode = 'a' if append else 'w'
        file = open(filename, mode, buffering=1)
        return partial(print, file=file)
    else:
        return partial(print, *default_prefix, file=default)


def basis_vector(dim, index):
    """Get D dimensional unit vector with only the i-th component being 1."""
    v = np.zeros(dim)
    v[index] = 1
    return v


def repeat(func, *args, **kwargs):
    """Call the function endlessly."""
    while True:
        yield func(*args, **kwargs)


def take(count, iterable):
    """Take count elements from iterable."""
    for i, v in zip(range(count), iterable):
        yield v


def project_to_plane(v, n):
    """Project v into the subspace defined by x∙n = 0."""
    return v - n * np.dot(v, n) / np.linalg.norm(n)


def is_int_vector(v):
    return all(np.round(v) == make_int_exact(v))


def get_bits(num):
    """Return tuple of indices corresponding to 1-bits."""
    return tuple(i for i in range(num.bit_length())
                 if num & (1 << i))


def default_column_label(index):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(alphabet[i] for i in get_bits(index))


def default_column_labels(dim):
    # TODO: assert dim=2**n
    return ['_'] + list(map(default_column_label, range(1, dim)))


class VectorMemory:

    """
    Remember vectors and return if they have been seen.

    Currently works only for int vectors.
    """

    def __init__(self):
        self.seen = set()

    def __call__(self, v):
        v = tuple(scale_to_int(v))
        if v in self.seen:
            return True
        self.seen.add(v)
        return False

    def add(self, *rows):
        for v in rows:
            self(v)
