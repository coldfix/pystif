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


def _name_list(s):
    try:
        return int(s)
    except ValueError:
        pass
    if path.exists(s):
        with open(s) as f:
            return f.read().split()
    return s.split()


class System:

    """
    IO utility for systems. Keeps track of column names.
    """

    def __init__(self, matrix=None, columns=None):
        self.matrix = matrix
        self.columns = columns

    @classmethod
    def load(cls, filename=None, *, default=sys.stdin, force=True):
        if not force:
            if not filename or filename == '-' or not path.exists(filename):
                return cls()
        return cls(*read_system(filename))

    @property
    def dim(self):
        return self.matrix.shape[1]

    @property
    def shape(self):
        return self.matrix.shape

    def __bool__(self):
        return self.matrix is not None

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
        return System(self.matrix[:,indices], columns), subdim

    def merge(self, other):
        if not self: return other
        if not other: return self
        assert self.columns and other.columns, \
            "Need to set column names for merge operation!"
        columns = self.columns[:]
        columns += [c for c in other.columns if c not in columns]
        col_idx = [columns.index(c) for c in other.columns]
        matrix = np.zeros((self.shape[0]+other.shape[0], len(columns)))
        matrix[:self.shape[0],:self.shape[1]] = self.matrix
        matrix[self.shape[0]:,col_idx] = other.matrix
        return self.__class__(matrix, columns)

    def _get_column_index(self, col):
        try:
            return int(col)
        except ValueError:
            return self.columns.index(col)

    def lp(self):
        """Get the LP."""
        return Problem(self.matrix)

    def prepare_for_projection(self, subspace):
        """
        Return a tuple ``(system, subspace_dimension)`` with the subspace
        occupying the columns with the lowest indices in the returned system.
        The ``subspace`` parameter can be either of:

            - integer  — subspace dimension, using the leftmost columns
            - filename — file containing the subspace column names
            - string   — string containing the subspace column names
        """
        subspace_columns = _name_list(subspace)
        if isinstance(subspace_columns, int):
            return self, subspace_columns
        return self.slice(subspace_columns, fill=True)


class SystemFile:

    """Sink for matrix files."""

    def __init__(self, filename=None, *,
                 default=sys.stdout, append=False, columns=None):
        self.columns = columns
        self.file_columns = columns
        self._seen = VectorMemory()
        self._slice = None
        self._started = False
        self._matrix = None
        if append:
            self._read_for_append(filename)
        self._print = print_to(filename, default=default, append=append)

    def _read_for_append(self, filename):
        system = System.load(filename, force=False)
        if system.matrix:
            self._matrix = system.matrix
            self._seen.add(*system.matrix)
            self._started = True
        if system.columns:
            self.file_columns = file_columns = system.columns
            if self.columns:
                self._slice = list(map(self.columns.index, file_columns))
            else:
                self.columns = file_columns

    def __call__(self, v):
        """Output the vector ``v``."""
        if self._seen(v):
            return
        if not self._started:
            if self.file_columns:
                self._print('#::', *self.file_columns)
            self._started = True
        if self._slice:
            v = v[self._slice]
        self._print(format_vector(v))


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


def default_column_labels(dim):
    return ['_'] + ['_'+str(i) for i in range(1, dim)]


def _column_label(index, varnames="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    return "".join(varnames[i] for i in get_bits(index))


def column_varname_labels(varnames):
    if isinstance(varnames, int):
        varnames = [chr(ord('A') + i) for i in range(varnames)]
    dim = 2**len(varnames)
    return ['_'] + [_column_label(i, varnames) for i in range(1, dim)]


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
