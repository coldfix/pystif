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
    with open(filename) as file:
        contents = remove_comments(file, comments.append)
        matrix = np.loadtxt(contents, ndmin=ndmin)
    cols = []
    for line in comments:
        detect_prefix(line.strip(), '::', lambda s: cols.extend(s.split()))
    return cols, matrix


class System:

    """
    IO utility for systems. Keeps track of column names.
    """

    def __init__(self, filename=None, *, read=True, write=False,
                 default=sys.stdout):
        self.seen = VectorMemory()
        self.matrix = None
        self.cols = None
        if read:
            if filename == '-' or not filename:
                read = False
            elif write and not path.exists(filename):
                read = False
        if read:
            self.cols, self.matrix = read_system(filename)
            if not self.cols:
                self.cols = None
            self.seen.add(*self.matrix)
        if write:
            if filename and filename != '-':
                self.file = open(filename, 'a' if read else 'w')
            else:
                self.file = default
        else:
            self.file = None

    # TODO: print only the tracked columns
    # TODO: print columns if not appending

    def add(self, v):
        """Output the vector ``v``."""
        if self.seen(v):
            return
        if self.matrix is None:
            self.matrix = np.array([v])
        else:
            self.matrix = np.vstack((self.matrix, v))
        if self.file:
            print(format_vector(v), file=self.file)

    def lp(self):
        """Get the LP."""
        return Problem(self.matrix)


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
