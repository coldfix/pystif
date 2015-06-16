"""
Utility functions that need Cython based optimization.
"""

import numpy as np
from array import array


cdef c_array.array int_array_template = array('i', [])
cdef c_array.array double_array_template = array('d', [])


def _as_np_array(x):
    """Create a numpy array from x suited for further processing."""
    return np.ascontiguousarray(x, np.float64)


def _as_matrix(x):
    x = _as_np_array(x)
    if len(x.shape) == 1:
        return np.array([x])
    return x


cdef str fmt_num(float num, float threshold=1e-10):
    cdef float r = round(num)
    if abs(num - r) < threshold:
        return "{:3}".format(int(r))
    return "{:22.15e}".format(num)


# Call me crazy for optimizing this function using cython, but this actually
# reduces the runtime of pystif.el_ineqs by about a factor 20.
def format_vector(v):
    """Convert vector to high-precision string, readable by np.loadtxt."""
    cdef double[:] c = double_view(v)
    return " ".join(fmt_num(x) for x in c)
