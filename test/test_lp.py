# encoding: utf-8

import unittest

from pystif.core.lp import Problem, UnboundedError, NofeasibleError
from numpy.testing import assert_array_equal, assert_almost_equal


class TestLP(unittest.TestCase):

    def test_add_rows(self):
        lp = Problem()
        lp.add_rows(10)
        self.assertEqual(lp.num_rows, 10)

    def test_add_cols(self):
        lp = Problem()
        lp.add_cols(10)
        self.assertEqual(lp.num_cols, 10)

    def test_add_row(self):
        lp = Problem()
        row = [1,2,3,4,5]
        lp.add_cols(len(row))
        lp.add_row(row)
        assert_array_equal(lp.get_row(0), row)
        with self.assertRaises(ValueError):
            lp.add_row([1])
        with self.assertRaises(IndexError):
            lp.get_row(1)

    def test_add_col(self):
        lp = Problem()
        col = [1,2,3,4,5]
        lp.add_rows(len(col))
        lp.add_col(col)
        assert_array_equal(lp.get_col(0), col)
        with self.assertRaises(ValueError):
            lp.add_col([1])
        with self.assertRaises(IndexError):
            lp.get_col(1)

    def test_get_matrix(self):
        rows = [[1,2,3,4,5],
                [6,7,8,9,0]]
        lp = Problem(rows)
        assert_array_equal(lp.get_matrix(), rows)

    def test_set_row_bnds(self):
        lp = Problem()
        lp.add_row()
        inf = float("inf")
        bounds_list = [
            (0, 2),         # DB
            (-inf, inf),    # FR
            (-inf, 0),      # UP
            (0, 0),         # FX
            (0, inf),       # LO
        ]
        for bnds in bounds_list:
            lp.set_row_bnds(0, *bnds)
            self.assertEqual(lp.get_row_bnds(0), bnds)

    def test_set_col_bnds(self):
        lp = Problem()
        lp.add_col()
        inf = float("inf")
        bounds_list = [
            (0, 2),         # DB
            (-inf, inf),    # FR
            (-inf, 0),      # UP
            (0, 0),         # FX
            (0, inf),       # LO
        ]
        for bnds in bounds_list:
            lp.set_col_bnds(0, *bnds)
            self.assertEqual(lp.get_col_bnds(0), bnds)

    def test_simple_optimize(self):
        lp = Problem()
        lp.add_col(lb=0)               #  0 ≤ x ≤ ∞
        lp.add_col(ub=2)               # -∞ ≤ y ≤ 2
        lp.add_row([+1, -1], ub=3)     #  x - y ≤ 3
        smin = lp.minimize([1, 2])     # min[ s = x + 2y ]
        smax = lp.maximize([1, 2])     # max[ s = x + 2y ]
        assert_almost_equal(smin, [0, -3])
        assert_almost_equal(smax, [5, +2])

    def test_optimize_unbounded(self):
        lp = Problem()
        lp.add_col(lb=0)               #  0 ≤ x ≤ ∞
        lp.add_col()                   # -∞ ≤ y ≤ ∞
        lp.add_row([+1, -1], ub=3)     #  x - y ≤ 3
        with self.assertRaises(UnboundedError):
            lp.maximize([1, 2])        # max[ s = x + 2y ]

    def test_optimize_nofeasible(self):
        lp = Problem()
        lp.add_col(lb=0)               #  0 ≤ x ≤  ∞
        lp.add_col(ub=-4)              # -∞ ≤ y ≤ -4
        lp.add_row([+1, -1], ub=3)     #  x - y ≤  3
        with self.assertRaises(NofeasibleError):
            lp.maximize([1, 2])        # max[ s = x + 2y ]

    def test_implies(self):
        L = [[0, 1,  0],   # x ≥ 0 (area left to y axis)
             [0, 0,  1],   # y ≥ 0 (area above x axis)
             [0, 1, -1]]   # x ≥ y (area under x=y axis)
        lp = Problem(L)
        self.assertTrue(lp.implies([0, 1, -1]))
        self.assertFalse(lp.implies([0, -1, 1]))

    def test_copy(self):
        a = Problem([0, 1, 1])  # x+y ≥ 0
        b = a.copy()
        a.add([0, -1,  0])      # x ≤ 0
        b.add([0,  0, -1])      # y ≤ 0
        self.assertTrue(a.implies([0, 0, 1]))   # y ≥ 0
        self.assertFalse(b.implies([0, 0, 1]))  # y ≥ 0
        self.assertFalse(a.implies([0, 1, 0]))  # x ≥ 0
        self.assertTrue(b.implies([0, 1, 0]))   # x ≥ 0


if __name__ == '__main__':
    unittest.main()
