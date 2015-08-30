import numpy as np


cdef extern from "math.h":
    double INF "INFINITY"
    double NAN "NAN"

cdef extern from "float.h":
    double DBL_MAX "DBL_MAX"


cdef int[:] int_array(int size)
cdef double[:] double_array(int size)


cdef inline double[:] double_view(x):
    """Get a memory view of sequence."""
    try:
        return x
    except (ValueError, TypeError):
        # ValueError: numpy dtype mismatch (e.g. int)
        # TypeError: x doesn't support buffer interface (e.g. list)
        return np.ascontiguousarray(x, np.float64)
