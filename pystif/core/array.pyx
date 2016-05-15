# cython: embedsignature=True
"""
Utility functions that need Cython based optimization.
"""

from __future__ import absolute_import

from array import array
from cpython cimport array as c_array
cimport cython
import numpy as np

from fractions import Fraction


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


cdef int[:] int_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(int_array_template, size, zero=False)


cdef double[:] double_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(double_array_template, size, zero=False)


# Call me crazy for optimizing this function using cython, but this actually
# reduces the runtime of pystif.el_ineqs by about a factor 20.
def format_vector(v):
    """Convert vector to high-precision string, readable by np.loadtxt."""
    cdef double[:] c = double_view(v)
    return " ".join(fmt_num(x) for x in c)



@cython.boundscheck(False)
@cython.wraparound(False)
def make_int_exact(v, double threshold=1e-10):
    """Replace numbers close to integers by the integers."""
    ret = np.empty(v.shape)
    cdef double[:] inp = double_view(v)
    cdef double[:] out = ret
    cdef double c, r
    cdef int i
    for i in range(inp.size):
        c = inp[i]
        r = round(c)
        if abs(c - r) < threshold:
            out[i] = r
        else:
            out[i] = c
    return ret


def scale_to_min(v, double threshold=1e-10):
    """Scale v such that its lowest non-zero component is one."""
    cdef double min = INF
    cdef double c, a
    for c in double_view(v):
        a = abs(c)
        if a > threshold and a < min:
            min = a
    return v/min


def scale_to_int(v, double threshold=1e-5, int max_factor=1000):
    """Scale v such that it has only integer components."""
    v = scale_to_min(v, threshold)
    cdef double c, amc
    cdef long m = 1
    for c in double_view(v):
        amc = abs(m*c)
        f = Fraction(amc).limit_denominator(max_factor//m + 1)
        if abs(f.numerator - f.denominator*amc) > threshold:
            return v
        m *= f.denominator
        if m > max_factor:
            return v
    return make_int_exact(v*m)
