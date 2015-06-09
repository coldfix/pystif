import numpy as np


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


def fmt_num(num):
    if round(num) == make_int_exact(num):
        return "{:3}".format(int(round(num)))
    return "{:22.15e}".format(num)


def format_vector(v):
    """Convert vector to high-precision string, readable by np.loadtxt."""
    return " ".join(map(fmt_num, v))


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


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def normalize(v):
    """Inplace normalization of coefficients."""
    # cancel gcd of coefficients
    if not v:
        return
    it = iter(v)
    for v in it:
        if v:
            div = abs(v)
            break
    for v in it:
        if v:
            div = gcd(div, abs(v))
            if div == 1:
                return v
    return v / div


def scale_to_int(v):
    """Scale v such that it has only integer components."""
    v = make_int_exact(v)
    v = scale_to_min(v)
    for c in range(1, 100):     # brute force:(
        cv = c*v
        r = np.round(cv)
        if all(r == make_int_exact(cv)):
            return normalize(r)
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
        if any(v != np.round(v)):
            return False
        v = tuple(int(c) for c in v)
        if v in self.seen:
            return True
        self.seen.add(v)
        return False
