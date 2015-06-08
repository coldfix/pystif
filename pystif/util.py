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


def project_to_plane(v, n):
    """Project v into the subspace defined by x∙n = 0."""
    return v - n * np.dot(v, n) / np.linalg.norm(n)


def format_vector(v):
    """Convert vector to high-precision string, readable by np.loadtxt."""
    return " ".join("{:22.15e}".format(c) for c in v)


@np.vectorize
def make_int_exact(c, threshold=1e-10):
    """Replace numbers close to integers by the integers."""
    r = round(c)
    if abs(c - r) < threshold:
        return r
    return c


def scale_to_int(v):
    """Scale v such that it has only integer components."""
    v = make_int_exact(v)
    for c in range(2, 100):     # brute force:(
        cv = c*v
        r = np.round(cv)
        if all(r == make_int_exact(cv)):
            return r
    return make_int_exact(v)
