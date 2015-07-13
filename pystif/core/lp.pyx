"""
Cython Wrapper for GLPK. For more information, see :class:`Problem`.
"""

cimport cython
import numpy as np

from . cimport glpk as glp
from .array cimport int_array, double_array, double_view, INF, NAN, DBL_MAX
from .array import _as_matrix
from .util import call


__all__ = [
    'MIN',
    'MAX',
    'PRIMAL',
    'DUAL',
    'OptimizeError',
    'UnknownError',
    'UnboundedError',
    'InfeasibleError',
    'NofeasibleError',
    'Problem',
]


cpdef enum:
    MIN = glp.MIN
    MAX = glp.MAX

cpdef enum:
    PRIMAL = glp.PRIMAL
    DUAL = glp.DUAL


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

    In general, input vectors q represent inequalities q∙x ≥ 0. There is a
    simple way to include an inhomogeneity in the vector q: Just set the
    bounds of the zero-th column to be exactly ``1``. Then the inequality
    represented by a vector q is:

            q₀ + q∙x ≥ 0

    In fact, this behaviour is enabled by default and all of the pystif
    modules assume equations in this form. For your own LPs this mode can be
    disabled by specifying ``const_col=False`` in the constructor.

    By default rows are lower-bounded by zero (see above) and columns are
    unbounded (i.e. variables may assume any value).
    """

    cdef glp.Prob* _lp

    def __cinit__(self, L=None, *,
                  int num_cols=0,
                  double lb_row=0, double ub_row=INF,
                  double lb_col=-INF, double ub_col=INF,
                  bint const_col=True):
        """
        Init system from constraint matrix L.

        You must either specify the constraint matrix or a positive value for
        the number of columns for ``const_col`` to take effect.
        """
        self._lp = glp.create_prob()
        if L is not None:
            L = _as_matrix(L)
            self.add(L, lb_row, ub_row, lb_col, ub_col, const_col=const_col)
        elif num_cols > 0:
            self.add_cols(num_cols, lb_col, ub_col, const_col=const_col)

    def __dealloc__(self):
        glp.delete_prob(self._lp)

    def copy(self):
        """Return an independent copy of this LP."""
        lp = Problem()
        glp.copy_prob(lp._lp, self._lp, glp.ON)
        return lp

    property num_rows:
        """Current number of rows."""
        def __get__(self):
            return glp.get_num_rows(self._lp)

    property num_cols:
        """Current number of cols."""
        def __get__(self):
            return glp.get_num_cols(self._lp)

    property shape:
        """Shape of the matrix you will get from get_matrix()."""
        def __get__(self):
            return self.num_rows, self.num_cols

    def add(self, L,
            double lb_row=0, double ub_row=INF,
            double lb_col=-INF, double ub_col=INF,
            *, bint embed=False, bint const_col=True):
        """
        Add the constraint matrix L∙x ≥ 0. Return the row index of the first
        added constraint.
        """
        L = _as_matrix(L)
        num_rows, num_cols = L.shape
        cdef int i
        cdef int s = self.add_rows(num_rows, lb_row, ub_row)
        if self.num_cols == 0:
            self.add_cols(num_cols, lb_col, ub_col, const_col=const_col)
        try:
            for i, row in enumerate(L):
                self.set_row(s+i, row, embed=embed)
            return s
        except:
            self.del_rows(range(s, s+num_rows))
            raise

    def add_row(self, coefs=None, double lb=0, double ub=INF, *, embed=False):
        """
        Add one row with specified bounds. If coefs is given, its size must be
        equal to the current number of cols. Return the row index.
        """
        cdef int i = self.add_rows(1, lb, ub)
        try:
            if coefs is not None:
                self.set_row(i, coefs, embed=embed)
            return i
        except:
            self.del_row(i)
            raise

    def add_col(self, coefs=None, double lb=-INF, double ub=INF, *, embed=False):
        """
        Add one col with specified bounds. If coefs is given, its size must be
        equal to the current number of rows.
        """
        cdef int i = self.add_cols(1, lb, ub, const_col=False)
        try:
            if coefs is not None:
                self.set_col(i, coefs, embed=embed)
            return i
        except:
            self.del_col(i)
            raise

    def add_rows(self, int num_rows, double lb=0, double ub=INF):
        """Add multiple rows and set their bounds."""
        cdef int s = glp.add_rows(self._lp, num_rows)-1
        try:
            self.set_row_bnds(range(s, s+num_rows), lb, ub)
            return s
        except:
            self.del_rows(range(s, s+num_rows))
            raise

    def add_cols(self, int num_cols, double lb=-INF, double ub=INF, *,
                 bint const_col=True):
        """Add multiple cols and set their bounds."""
        if num_cols <= 0:
            raise ValueError("Invalid number of columns: {}"
                             .format(num_cols))
        cdef int s = glp.add_cols(self._lp, num_cols)-1
        try:
            self.set_col_bnds(range(s, s+num_cols), lb, ub)
            if const_col and self.num_cols == num_cols:
                self.set_col_bnds(0, 1, 1)
            return s
        except:
            self.del_cols(range(s, s+num_cols))
            raise

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def del_rows(self, rows):
        cdef int[:] buf = np.array(rows, dtype=np.intc)
        cdef int i
        for i in range(buf.size):
            self._check_row_index(i)
            buf[i] += 1
        glp.del_rows(self._lp, buf.size, &buf[0]-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def del_cols(self, cols):
        cdef int[:] buf = np.array(cols, dtype=np.intc)
        cdef int i
        for i in range(buf.size):
            self._check_col_index(i)
            buf[i] += 1
        glp.del_cols(self._lp, buf.size, &buf[0]-1)

    def del_row(self, int row):
        """Delete one row."""
        self._check_row_index(row)
        row += 1
        glp.del_rows(self._lp, 1, &row-1)

    def del_col(self, int col):
        """Delete one col."""
        self._check_col_index(col)
        col += 1
        glp.del_cols(self._lp, 1, &col-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_row(self, int row, coefs, *, embed=False):
        """Set coefficients of one row."""
        self._check_row_index(row)
        cdef int num_cols = self.num_cols
        # TODO: more efficient to forward only the nonzero components?
        cdef double[:] val = double_view(coefs)
        cdef int   [:] ind = np.arange(1, val.size+1, dtype=np.intc)
        self._check_row_size(val.size, embed)
        glp.set_mat_row(self._lp, row+1, val.size, &ind[0]-1, &val[0]-1)

    def set_col(self, int col, coefs, *, embed=False):
        """Set coefficients of one col."""
        self._check_col_index(col)
        # TODO: more efficient to forward only the nonzero components?
        cdef double[:] val = double_view(coefs)
        cdef int   [:] ind = np.arange(1, val.size+1, dtype=np.intc)
        self._check_col_size(val.size, embed)
        glp.set_mat_col(self._lp, col+1, val.size, &ind[0]-1, &val[0]-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_row(self, int row):
        """Get coefficient vector of one row."""
        self._check_row_index(row)
        cdef int num_cols = self.num_cols
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
        self._check_col_index(col)
        cdef int num_rows = self.num_rows
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
    def get_matrix(self):
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
            rows = (rows,)
        cdef int vartype = get_vartype(lb, ub)
        cdef int row
        for row in rows:
            self._check_row_index(row)
            glp.set_row_bnds(self._lp, row+1, vartype, lb, ub)

    def set_col_bnds(self, cols, double lb, double ub):
        """Set bounds of the specified col."""
        if isinstance(cols, int):
            cols = (cols,)
        cdef int vartype = get_vartype(lb, ub)
        cdef int col
        for col in cols:
            self._check_col_index(col)
            glp.set_col_bnds(self._lp, col+1, vartype, lb, ub)

    def get_row_bnds(self, int row):
        """Get bounds (lb, ub) of the specified row."""
        self._check_row_index(row)
        cdef double lb = glp.get_row_lb(self._lp, row+1)
        cdef double ub = glp.get_row_ub(self._lp, row+1)
        return (-INF if lb == -DBL_MAX else lb,
                +INF if ub == +DBL_MAX else ub)

    def get_col_bnds(self, int col):
        """Get bounds (lb, ub) of the specified col."""
        self._check_col_index(col)
        cdef double lb = glp.get_col_lb(self._lp, col+1)
        cdef double ub = glp.get_col_ub(self._lp, col+1)
        return (-INF if lb == -DBL_MAX else lb,
                +INF if ub == +DBL_MAX else ub)

    cdef _check_row_size(self, int size, embed, str name='row'):
        if size == self.num_cols:
            return
        if embed and size >= 0 and size < self.num_cols:
            return
        raise ValueError("Expecting {} size {}, got {}."
                         .format(name, self.num_cols, size))

    cdef _check_col_size(self, int size, embed, str name='col'):
        if size == self.num_rows:
            return
        if embed and size >= 0 and size < self.num_rows:
            return
        raise ValueError("Expecting {} size {}, got {}."
                         .format(name, self.num_rows, size))

    cdef _check_row_index(self, int row):
        if row < 0 or row >= self.num_rows:
            raise IndexError("Invalid row = {}. Expecting 0 ≤ row < {}."
                             .format(row, self.num_rows))

    cdef _check_col_index(self, int col):
        if col < 0 or col >= self.num_cols:
            raise IndexError("Invalid col = {}. Expecting 0 ≤ col < {}."
                             .format(col, self.num_cols))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_objective(self, coefs, int sense=glp.MIN, *, embed=False):
        """Set objective coefficients and direction."""
        cdef double[:] buf = double_view(coefs)
        cdef int col
        self._check_row_size(buf.size, embed, "objective")
        glp.set_obj_dir(self._lp, sense)
        for col in range(buf.size):
            glp.set_obj_coef(self._lp, col+1, buf[col])
        for col in range(buf.size, self.num_cols):
            glp.set_obj_coef(self._lp, col+1, 0)

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

    def optimize(self, objective, sense=glp.MIN, *, embed=False):
        """Optimize objective and return solution as numpy array."""
        self.set_objective(objective, sense, embed=embed)
        self.simplex()
        return self.get_prim_solution()

    def minimize(self, objective, *, embed=False):
        return self.optimize(objective, glp.MIN, embed=embed)

    def maximize(self, objective, *, embed=False):
        return self.optimize(objective, glp.MAX, embed=embed)

    def get_objective_value(self):
        """Get value of objective achieved in last optimization task."""
        return glp.get_obj_val(self._lp)

    def has_optimal_solution(self, objective, sense=glp.MIN, *, embed=False):
        """Check if the system has an optimal solution."""
        try:
            self.set_objective(objective, sense, embed=embed)
            self.simplex(sense)
            return True
        except (UnboundedError, InfeasibleError, NofeasibleError):
            return False

    def implies(self, L, *, embed=False, threshold=1e-14):
        """
        Check if the constraint matrix L∙x ≥ 0 is redundant, i.e. each point
        in the polytope specified by the LP satisfies the constraints in L.

        ``L`` can either be a matrix or a single row vector.
        """
        def _implies(q):
            return (self.has_optimal_solution(q, embed=embed) and
                    self.get_objective_value() >= -threshold)
        return all(map(_implies, _as_matrix(L)))

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


class Minimize:

    """
    Remove redundant inequalities from a system of linear inequalities. Be
    warned, this is a very slow process.
    """

    # callbacks to provide UI status information
    cb_start = None
    cb_step = None
    cb_stop = None

    def minimize(self, rows):
        call(self.cb_start, rows)
        ret = []
        lp = Problem(rows)
        for idx in range(lp.num_rows-1, -1, -1):
            row = rows[idx]
            call(self.cb_step, idx, ret)
            lp.del_row(idx)
            if not lp.implies(row):
                lp.add_row(row)
                ret.append(row)
        call(self.cb_stop, ret)
        return ret
