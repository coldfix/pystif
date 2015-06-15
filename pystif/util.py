import numpy as np
import sys
from functools import partial
from .core.lp import format_vector


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


@np.vectorize
def make_int_exact(c, threshold=1e-10):
    """Replace numbers close to integers by the integers."""
    r = round(c)
    if abs(c - r) < threshold:
        return r
    return c


def scale_to_min(v):
    """Scale v such that its lowest non-zero component is one."""
    inf = float('inf')
    return v / min((abs(c) if c != 0 else inf for c in v), default=1)


def scale_to_int(v):
    """Scale v such that it has only integer components."""
    v = make_int_exact(v)
    v = scale_to_min(v)
    for c in range(1, 100):     # brute force:(
        cv = c*v
        r = np.round(cv)
        if all(r == make_int_exact(cv)):
            return r
    return make_int_exact(v)


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
