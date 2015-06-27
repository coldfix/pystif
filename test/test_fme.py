import unittest

import numpy as np
from numpy.testing import assert_array_equal

from pystif.core.fme import FME


rows = np.array([
    [ 0,  0,  1,  1,  0],
    [ 0,  1, -1, -6,  0],
    [ 0,  2,  2,  3,  0],
    [ 0,  3, -2,  9,  0],
])


def assert_arrays_equal(a, b):
    for x, y in zip(a, b):
        assert_array_equal(x, y)


class TestFME(unittest.TestCase):

    def test_partition(self):

        fme = FME()

        assert_arrays_equal(
            fme.partition_by_column_sign(rows, 0),
            (rows, [], []))

        assert_arrays_equal(
            fme.partition_by_column_sign(rows, 1),
            (rows[[0]], rows[[1,2,3]], []))

        assert_arrays_equal(
            fme.partition_by_column_sign(rows, 2),
            ([], rows[[0,2]], rows[[1,3]]))

        assert_arrays_equal(
            fme.partition_by_column_sign(rows, 3),
            ([], rows[[0,2,3]], rows[[1]]))

        assert_arrays_equal(
            fme.partition_by_column_sign(rows, 4),
            (rows, [], []))

    def test_eliminate_pair(self):

        fme = FME()

        assert_array_equal(
            fme.eliminate_column_from_pair(rows[0], rows[1], 2),
            [0, 1, -5, 0])

        assert_array_equal(
            fme.eliminate_column_from_pair(rows[0], rows[1], 3),
            [0, 1, 5, 0])

        assert_array_equal(
            fme.eliminate_column_from_pair(rows[2], rows[1], 3),
            [0, 5, 3, 0])

        assert_array_equal(
            fme.eliminate_column_from_pair(rows[3], rows[1], 3),
            [0, 9, -7, 0])

    def test_eliminate_column(self):

        fme = FME()

        assert_array_equal(
            fme.eliminate_column_from_system(rows, 0),
            rows[:,1:])

        assert_array_equal(
            fme.eliminate_column_from_system(rows, 1),
            [np.delete(rows[0], 1)])

        assert_array_equal(
            fme.eliminate_column_from_system(rows, 3),
            [[0, 1, 5, 0],
             [0, 5, 3, 0],
             [0, 9, -7, 0]])

        assert_array_equal(
            fme.eliminate_column_from_system(rows, 4),
            rows[:,:-1])


if __name__ == '__main__':
    unittest.main()
