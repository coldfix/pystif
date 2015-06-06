# encoding: utf-8

import unittest
from numpy.testing import assert_array_equal, assert_almost_equal

from pystif.core import it

import numpy as np


class TestIT(unittest.TestCase):

    # Note that these tests are suboptimal since they test for an exact
    # implementation rather than behaviour. But at least they will make it
    # visible if something is drastically changed by accident.

    def test_elemental_inequalities_1(self):
        expected = [
            [0,  1],
        ]
        assert_array_equal(list(it.elemental_inequalities(1, int)), expected)

    def test_elemental_inequalities_2(self):
        expected = [
            [0,  0, -1,  1],
            [0, -1,  0,  1],
            [0,  1,  1, -1],
        ]
        assert_array_equal(list(it.elemental_inequalities(2, int)), expected)

    def test_elemental_inequalities_2(self):
        expected = [
            [0,  0,  0,  0,  0,  0, -1,  1],
            [0,  0,  0,  0,  0, -1,  0,  1],
            [0,  0,  0, -1,  0,  0,  0,  1],
            [0,  1,  1, -1,  0,  0,  0,  0],
            [0,  0,  0,  0, -1,  1,  1, -1],
            [0,  1,  0,  0,  1, -1,  0,  0],
            [0,  0, -1,  1,  0,  0,  1, -1],
            [0,  0,  1,  0,  1,  0, -1,  0],
            [0, -1,  0,  1,  0,  1,  0, -1],
        ]
        assert_array_equal(list(it.elemental_inequalities(3, int)), expected)

    def test_cyclic_cca_causal_constraints_1d(self):
        # TODO
        pass

    def test_mutual_independence_constraints_3(self):
        expected = [
            [0, -1, -1,  0, -1,  0,  0,  1],
        ]
        acquired = it.mutual_independence_constraints(range(3), 3, int)
        assert_array_equal(list(acquired), expected)

if __name__ == '__main__':
    unittest.main()
