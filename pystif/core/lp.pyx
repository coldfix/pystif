
cimport cython
cimport glpk as glp

from cpython cimport array as c_array
from array import array
from itertools import repeat, starmap

import numpy as np

cdef extern from "math.h":
    double INF "INFINITY"
    double NAN "NAN"

cdef extern from "float.h":
    double DBL_MAX "DBL_MAX"


cpdef enum:
    MIN = glp.MIN
    MAX = glp.MAX

cpdef enum:
    PRIMAL = glp.PRIMAL
    DUAL = glp.DUAL


cdef c_array.array int_array_template = array('i', [])
cdef c_array.array double_array_template = array('d', [])

cdef int[:] int_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(int_array_template, size, zero=False)

cdef double[:] double_array(int size):
    """Create a fixed size buffer."""
    return c_array.clone(double_array_template, size, zero=False)

cdef double[:] double_view(x):
    """Get a memory view of sequence."""
    try:
        return x
    except (ValueError, TypeError):
        # ValueError: numpy dtype mismatch (e.g. int)
        # TypeError: x doesn't support buffer interface (e.g. list)
        return np.ascontiguousarray(x, np.float64)


def _as_np_array(x):
    """Create a numpy array from x suited for further processing."""
    return np.ascontiguousarray(x, np.float64)


cdef int get_vartype(double lb, double ub):
    if lb == ub:
        return glp.FX
    if lb == -INF and ub == +INF:
        return glp.FR
    if lb == -INF and ub <  +INF:
        return glp.UP
    if lb >  -INF and ub == +INF:
        return glp.LO
    return glp.DB


class OptimizeError(RuntimeError):
    """Any error that occured during optimization."""

class UnknownError(OptimizeError):
    """Unknown error in glpk."""

class UnboundedError(OptimizeError):
    """Objective is unbounded."""

class InfeasibleError(OptimizeError):
    """Current solution is infeasible."""

class NofeasibleError(OptimizeError):
    """No feasible solution exists."""


cdef class Problem:

    """
    Simple wrapper for the GLPK problem API.

    The wrapper API exposes zero-based indices!

    The accessor methods return numpy arrays where applicable. As input
    vectors you should use numpy arrays or other objects that export the
    buffer API to get the best performance.

    The wrapper is not complete. I'm adding what I need as I go forward.
    """

    cdef glp.Prob* _lp

    def __cinit__(self, int num_cols=0):
        """Init empty system and optionally set number of columns."""
        self._lp = glp.create_prob()
        if num_cols > 0:
            self.add_cols(num_cols)

    def __dealloc__(self):
        glp.delete_prob(self._lp)

    property num_rows:
        """Current number of rows."""
        def __get__(self):
            return glp.get_num_rows(self._lp)

    property num_cols:
        """Current number of cols."""
        def __get__(self):
            return glp.get_num_cols(self._lp)

    @classmethod
    def from_matrix(cls, matrix,
                    double lb_row=-INF, double ub_row=INF,
                    double lb_col=-INF, double ub_col=INF):
        """Create a Problem from a matrix."""
        matrix = _as_np_array(matrix)
        lp = cls()
        lp.add_cols(matrix.shape[1], lb_col, ub_col)
        lp.add_matrix(matrix, lb_row, ub_row)
        return lp

    def add_matrix(self, matrix, double lb_row=-INF, double ub_row=INF):
        """Add matrix."""
        matrix = _as_np_array(matrix)
        num_rows, num_cols = matrix.shape
        if self.num_cols == 0:
            self.add_cols(num_cols)
        start = self.add_rows(num_rows, lb_row, ub_row)
        for i, row in enumerate(matrix):
            self.set_row(start+i, row)

    def add_row(self, coefs=None, double lb=-INF, double ub=INF):
        """
        Add one row with specified bounds. If coefs is given, its size must be
        equal to the current number of cols.
        """
        cdef int i = self.add_rows(1, lb, ub)
        if coefs is not None:
            try:
                self.set_row(i, coefs)
            except ValueError:
                self.del_row(i)
                raise
        return i

    def add_col(self, coefs=None, double lb=-INF, double ub=INF):
        """
        Add one col with specified bounds. If coefs is given, its size must be
        equal to the current number of rows.
        """
        cdef int i = self.add_cols(1, lb, ub)
        if coefs is not None:
            try:
                self.set_col(i, coefs)
            except ValueError:
                self.del_col(i)
                raise
        return i

    def add_rows(self, int num_rows, double lb=-INF, double ub=INF):
        """Add multiple rows and set their bounds."""
        cdef int s = glp.add_rows(self._lp, num_rows)-1
        cdef int i
        for i in range(s, s+num_rows):
            self.set_row_bnds(i, lb, ub)
        return s

    def add_cols(self, int num_cols, double lb=-INF, double ub=INF):
        """Add multiple cols and set their bounds."""
        cdef int s = glp.add_cols(self._lp, num_cols)-1
        cdef int i
        for i in range(s, s+num_cols):
            self.set_col_bnds(i, lb, ub)
        return s

    def del_row(self, int row):
        """Delete one row."""
        row += 1
        glp.del_rows(self._lp, 1, &row-1)

    def del_col(self, int col):
        """Delete one col."""
        col += 1
        glp.del_cols(self._lp, 1, &col-1)

    def set_row(self, int row, coefs):
        """Set coefficients of one row."""
        cdef int num_rows = self.num_rows
        cdef int num_cols = self.num_cols
        if row < 0 or row >= num_rows:
            raise ValueError("Row {} is out of range (0, {})."
                             .format(row, num_rows))
        # TODO: more efficient to forward only the nonzero components?
        cdef int   [:] ind = np.arange(1, num_cols+1, dtype=np.intc)
        cdef double[:] val = double_view(coefs)
        if val.size != num_cols:
            raise ValueError("Row size must be {}, got {}."
                             .format(num_cols, val.size))
        glp.set_mat_row(self._lp, row+1, num_cols, &ind[0]-1, &val[0]-1)

    def set_col(self, int col, coefs):
        """Set coefficients of one col."""
        cdef int num_rows = self.num_rows
        cdef int num_cols = self.num_cols
        if col < 0 or col >= num_cols:
            raise ValueError("Col {} is out of range (0, {})."
                             .format(col, num_cols))
        # TODO: more efficient to forward only the nonzero components?
        cdef int   [:] ind = np.arange(1, num_rows+1, dtype=np.intc)
        cdef double[:] val = double_view(coefs)
        if val.size != num_rows:
            raise ValueError("Col size must be {}, got {}."
                             .format(num_rows, val.size))
        glp.set_mat_col(self._lp, col+1, num_rows, &ind[0]-1, &val[0]-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_row(self, int row):
        """Get coefficient vector of one row."""
        cdef int num_rows = self.num_rows
        cdef int num_cols = self.num_cols
        if row < 0 or row >= num_rows:
            raise ValueError("Row {} is out of range (0, {})."
                             .format(row, num_rows))
        cdef int   [:] ind = int_array(num_cols)
        cdef double[:] val = double_array(num_cols)
        cdef int nz = glp.get_mat_row(self._lp, row+1, &ind[0]-1, &val[0]-1)
        ret = np.zeros(num_cols, np.float64, "c")
        cdef double[:] buf = ret
        cdef int i
        for i in range(nz):
            buf[ind[i]-1] = val[i]
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_col(self, int col):
        """Get coefficient vector of one col."""
        cdef int num_rows = self.num_rows
        cdef int num_cols = self.num_cols
        if col < 0 or col >= num_cols:
            raise ValueError("Col {} is out of range (0, {})."
                             .format(col, num_cols))
        cdef int   [:] ind = int_array(num_rows)
        cdef double[:] val = double_array(num_rows)
        cdef int nz = glp.get_mat_col(self._lp, col+1, &ind[0]-1, &val[0]-1)
        ret = np.zeros(num_rows, np.float64, "c")
        cdef double[:] buf = ret
        cdef int i
        for i in range(nz):
            buf[ind[i]-1] = val[i]
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_mat(self):
        """Get coefficient matrix."""
        cdef int num_rows = self.num_rows
        cdef int num_cols = self.num_cols
        cdef int   [:] ind = int_array(num_cols)
        cdef double[:] val = double_array(num_cols)
        cdef double[:] buf
        cdef int i, j, nz
        ret = np.zeros((num_rows, num_cols), np.float64, "c")
        for i in range(num_rows):
            nz = glp.get_mat_row(self._lp, i+1, &ind[0]-1, &val[0]-1)
            buf = ret[i]
            for j in range(nz):
                buf[ind[j]-1] = val[j]
        return ret

    def set_row_bnds(self, rows, double lb, double ub):
        """Set bounds of the specified row(s)."""
        if isinstance(rows, int):
            rows = [rows]
        cdef int vartype = get_vartype(lb, ub)
        cdef int row
        for row in rows:
            glp.set_row_bnds(self._lp, row+1, vartype, lb, ub)

    def set_col_bnds(self, cols, double lb, double ub):
        """Set bounds of the specified col."""
        if isinstance(cols, int):
            cols = [cols]
        cdef int vartype = get_vartype(lb, ub)
        cdef int col
        for col in cols:
            glp.set_col_bnds(self._lp, col+1, vartype, lb, ub)

    def get_row_bnds(self, int row):
        """Get bounds (lb, ub) of the specified row."""
        cdef double lb = glp.get_row_lb(self._lp, row+1)
        cdef double ub = glp.get_row_ub(self._lp, row+1)
        return (-INF if lb == -DBL_MAX else lb,
                +INF if ub == +DBL_MAX else ub)

    def get_col_bnds(self, int col):
        """Get bounds (lb, ub) of the specified col."""
        cdef double lb = glp.get_col_lb(self._lp, col+1)
        cdef double ub = glp.get_col_ub(self._lp, col+1)
        return (-INF if lb == -DBL_MAX else lb,
                +INF if ub == +DBL_MAX else ub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_objective(self, coefs, int sense=glp.MIN):
        """Set objective coefficients and direction."""
        cdef int num_cols = self.num_cols
        cdef double[:] buf = double_view(coefs)
        cdef int col
        if buf.size != num_cols:
            raise ValueError("Expecting objective size {}, got {}."
                             .format(num_cols, buf.size))
        glp.set_obj_dir(self._lp, sense)
        for col in range(num_cols):
            glp.set_obj_coef(self._lp, col+1, buf[col])

    def simplex(self, int method=glp.PRIMAL):
        """Perform simplex algorithm."""
        cdef glp.SmCp parm
        glp.init_smcp(&parm)
        parm.msg_lev = glp.MSG_ERR
        parm.meth = method
        glp.std_basis(self._lp)
        cdef int status = glp.simplex(self._lp, &parm)
        if status != 0:
            raise RuntimeError("Error in glp.simplex: {}".format(status))
        cdef int prim = glp.get_prim_stat(self._lp)
        cdef int dual = glp.get_dual_stat(self._lp)
        if method == glp.DUAL:
            prim, dual = dual, prim
        if prim == glp.FEAS and dual == glp.FEAS:
            return
        if prim == glp.FEAS and dual == glp.NOFEAS:
            raise UnboundedError
        if prim == glp.INFEAS:
            raise InfeasibleError
        if prim == glp.NOFEAS:
            raise NofeasibleError
        raise UnknownError

    def optimize(self, objective, sense=glp.MIN):
        """Optimize objective and return solution as numpy array."""
        self.set_objective(objective, sense)
        self.simplex()
        return self.get_prim_solution()

    def minimize(self, objective):
        return self.optimize(objective, glp.MIN)

    def maximize(self, objective):
        return self.optimize(objective, glp.MAX)

    def get_objective_value(self):
        """Get value of objective achieved in last optimization task."""
        return glp.get_obj_val(self._lp)

    def has_optimal_solution(self, objective, sense=glp.MIN):
        """Check if the system has an optimal solution."""
        try:
            self.optimize(objective, sense)
            return True
        except (UnboundedError, InfeasibleError, NofeasibleError):
            return False

    def implies(self, constraint, bound=0, sense=glp.MIN):
        """
        Check if the inequality represented by the constraint vector C and
        bound b is redundant.

        The direction of the inequality depends on the value of ``sense``:

            - C∙x ≥ b   if sense=lp.MIN (default)
            - C∙x ≤ b   if sense=lp.MAX

        For convenience, ``constraint`` can be a matrix containing multiple
        constraint queries. In this case ``bound`` and ``sense`` are allowed
        to be vectors (but don't need to be).
        """
        def _implies(c, b, s):
            if not self.has_optimal_solution(c, s):
                return False
            obj_val = self.get_objective_value()
            return obj_val <= b if s == glp.MIN else obj_val >= b
        constraint = _as_np_array(constraint)
        if len(constraint.shape) == 1:
            constraint = [constraint]
        if isinstance(bound, (float, int)):
            bound = repeat(bound)
        if isinstance(sense, int):
            sense = repeat(sense)
        return all(starmap(_implies, zip(constraint, bound, sense)))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_prim_solution(self):
        """Return primal solution as numpy array."""
        ret = np.empty(self.num_cols, np.float64, "c")
        cdef double[:] buf = ret
        cdef int col
        for col in range(self.num_cols):
            buf[col] = glp.get_col_prim(self._lp, col+1)
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_dual_solution(self):
        """Return dual solution as numpy array."""
        ret = np.empty(self.num_rows, np.float64, "c")
        cdef double[:] buf = ret
        cdef int row
        for row in range(self.num_rows):
            buf[row] = glp.get_row_dual(self._lp, row+1)
        return ret
