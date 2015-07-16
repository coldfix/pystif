from cpython cimport array as c_array
import numpy as np


cdef extern from "math.h":
    double INF "INFINITY"
    double NAN "NAN"

cdef extern from "float.h":
    double DBL_MAX "DBL_MAX"


cdef c_array.array int_array_template
cdef c_array.array double_array_template


cdef inline int[:] int_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(int_array_template, size, zero=False)


cdef inline double[:] double_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(double_array_template, size, zero=False)


cdef inline double[:] double_view(x):
    """Get a memory view of sequence."""
    try:
        return x
    except (ValueError, TypeError):
        # ValueError: numpy dtype mismatch (e.g. int)
        # TypeError: x doesn't support buffer interface (e.g. list)
        return np.ascontiguousarray(x, np.float64)
