# encoding: utf-8

import unittest

from pystif.core.lp import Problem, UnboundedError, NofeasibleError
from numpy.testing import assert_array_equal, assert_almost_equal


class TestLP(unittest.TestCase):

    def setUp(self):
        self.lp = Problem()

    def test_add_rows(self):
        self.lp.add_rows(10)
        self.assertEqual(self.lp.num_rows, 10)

    def test_add_cols(self):
        self.lp.add_cols(10)
        self.assertEqual(self.lp.num_cols, 10)

    def test_add_row(self):
        row = [1,2,3,4,5]
        self.lp.add_cols(len(row))
        self.lp.add_row(row)
        assert_array_equal(self.lp.get_row(0), row)
        with self.assertRaises(ValueError):
            self.lp.add_row([1])
        with self.assertRaises(ValueError):
            self.lp.get_row(1)

    def test_add_col(self):
        col = [1,2,3,4,5]
        self.lp.add_rows(len(col))
        self.lp.add_col(col)
        assert_array_equal(self.lp.get_col(0), col)
        with self.assertRaises(ValueError):
            self.lp.add_col([1])
        with self.assertRaises(ValueError):
            self.lp.get_col(1)

    def test_get_mat(self):
        rows = [[1,2,3,4,5],
                [6,7,8,9,0]]
        self.lp.add_cols(len(rows[0]))
        self.lp.add_row(rows[0])
        self.lp.add_row(rows[1])
        assert_array_equal(self.lp.get_mat(), rows)

    def test_set_row_bnds(self):
        self.lp.add_row()
        inf = float("inf")
        bounds_list = [
            (0, 2),         # DB
            (-inf, inf),    # FR
            (-inf, 0),      # UP
            (0, 0),         # FX
            (0, inf),       # LO
        ]
        for bnds in bounds_list:
            self.lp.set_row_bnds(0, *bnds)
            self.assertEqual(self.lp.get_row_bnds(0), bnds)

    def test_set_col_bnds(self):
        self.lp.add_col()
        inf = float("inf")
        bounds_list = [
            (0, 2),         # DB
            (-inf, inf),    # FR
            (-inf, 0),      # UP
            (0, 0),         # FX
            (0, inf),       # LO
        ]
        for bnds in bounds_list:
            self.lp.set_col_bnds(0, *bnds)
            self.assertEqual(self.lp.get_col_bnds(0), bnds)

    def test_simple_optimize(self):
        self.lp.add_col(lb=0)               #  0 ≤ x ≤ ∞
        self.lp.add_col(ub=2)               # -∞ ≤ y ≤ 2
        self.lp.add_row([+1, -1], ub=3)     #  x - y ≤ 3
        smin = self.lp.minimize([1, 2])     # min[ s = x + 2y ]
        smax = self.lp.maximize([1, 2])     # max[ s = x + 2y ]
        assert_almost_equal(smin, [0, -3])
        assert_almost_equal(smax, [5, +2])

    def test_optimize_unbounded(self):
        self.lp.add_col(lb=0)               #  0 ≤ x ≤ ∞
        self.lp.add_col()                   # -∞ ≤ y ≤ ∞
        self.lp.add_row([+1, -1], ub=3)     #  x - y ≤ 3
        with self.assertRaises(UnboundedError):
            self.lp.maximize([1, 2])        # max[ s = x + 2y ]

    def test_optimize_nofeasible(self):
        self.lp.add_col(lb=0)               #  0 ≤ x ≤  ∞
        self.lp.add_col(ub=-4)              # -∞ ≤ y ≤ -4
        self.lp.add_row([+1, -1], ub=3)     #  x - y ≤  3
        with self.assertRaises(NofeasibleError):
            self.lp.maximize([1, 2])        # max[ s = x + 2y ]


if __name__ == '__main__':
    unittest.main()
