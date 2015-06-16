import numpy as np

from cpython cimport array as c_array


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
