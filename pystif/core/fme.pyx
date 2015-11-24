
import itertools
from math import gcd

import numpy as np

from .lp import Problem
from .util import call


class FME:

    """
    Tools for performing a Fourier-Motzkin-Elimination on systems of linear
    inequalities. Inequalities must be specified as coefficient vectors q
    taking the form q∙(1,x) ≥ 0.

    This is kept as a class mainly to allow polymorphism (e.g. override
    column heuristic or redundancy check) — but also to allow adding output
    functions.
    """

    cb_start = None
    cb_step = None
    cb_stop = None

    def partition_by_column_sign(self, rows, int col):
        """
        Partition a system of inequalities according to the sign of the column
        coefficient.
        """
        cdef int val
        zero, pos, neg = [], [], []
        for row in rows:
            val = row[col]
            if val == 0:
                zero.append(row)
            elif val < 0:
                neg.append(row)
            elif val > 0:
                pos.append(row)
        return zero, pos, neg

    def eliminate_column_from_pair(self, pos_row, neg_row, int col):
        """
        Eliminate a column from a pairing of inequalities with opposite sign
        in the specified column.
        """
        pos_coef = pos_row[col]
        neg_coef = neg_row[col]
        div = gcd(pos_coef, -neg_coef)
        scaled = (pos_row * (-neg_coef // div) +
                  neg_row * (+pos_coef // div))
        # TODO: normalize result
        return np.delete(scaled, col)

    def eliminate_column_from_system(self, rows, int col):
        """Eliminate a column from a system of inequalities."""
        zero, pos, neg = self.partition_by_column_sign(rows, col)
        call(self.cb_step, rows, col, zero, pos, neg)
        ret = [np.delete(row, col) for row in zero]
        if ret:
            lp = Problem(ret)
        else:
            lp = Problem(num_cols=len(rows[0])-1)
        for p, n in itertools.product(pos, neg):
            row = self.eliminate_column_from_pair(p, n, col)
            if not lp.implies(row):
                lp.add(row)
                ret.append(row)
        return ret

    def get_column_rank(self, rows, int col):
        """
        This function is used as heuristic for choosing which column to
        eliminate next by taking the column with the lowest associated rank
        value. Concretely, it currently calculates the maximum number of added
        rows if eliminating the specified column.
        """
        cdef int pos = 0
        cdef int neg = 0
        for row in rows:
            val = row[col]
            if val > 0:
                pos += 1
            elif val < 0:
                neg += 1
        return (pos*neg) - (pos+neg)

    def choose_next_column(self, rows, cols_to_eliminate):
        """Heuristic for choosing the next column to eliminate."""
        return min(((col, self.get_column_rank(rows, col))
                    for col in cols_to_eliminate),
                   key=lambda cr: cr[1])[0]

    def eliminate_columns_from_system(self, rows, cols_to_eliminate):
        """Eliminate columns with the specified indices from the system."""
        cdef int col
        call(self.cb_start, rows, cols_to_eliminate)
        while cols_to_eliminate:
            col = self.choose_next_column(rows, cols_to_eliminate)
            rows = self.eliminate_column_from_system(rows, col)
            cols_to_eliminate = [
                c-1 if c > col else c
                for c in cols_to_eliminate
                if c != col
            ]
        call(self.cb_stop, rows)
        return rows

    def solve_to(self, rows, int solve_to):
        """
        Eliminate all columns with indices higher than ``solve_to`` from
        the system.
        """
        num_cols = len(rows[0])
        cols_to_eliminate = list(range(solve_to, num_cols))
        return self.eliminate_columns_from_system(rows, cols_to_eliminate)
