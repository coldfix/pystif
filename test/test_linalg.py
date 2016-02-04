import unittest

import numpy as np
from numpy.testing import assert_allclose

from pystif.core import linalg


# TODO: use deterministic data for these tests… 

class TestLA(unittest.TestCase):

    def test_ptrace(self):
        kron = linalg.kron
        ptrace = linalg.ptrace

        def random_matrix():
            rho = linalg.random_quantum_state(4).reshape((2, 2))
            return rho / rho.trace()

        rho_A = random_matrix()
        rho_B = random_matrix()
        rho_C = random_matrix()

        rho_AB = kron(rho_A, rho_B)
        rho_BC = kron(rho_B, rho_C)
        rho_AC = kron(rho_A, rho_C)
        rho_ABC = kron(rho_A, rho_B, rho_C)

        dims = (2, 2, 2)
        assert_allclose(rho_AB, ptrace(rho_ABC, dims, 2))
        assert_allclose(rho_BC, ptrace(rho_ABC, dims, 0))
        assert_allclose(rho_AC, ptrace(rho_ABC, dims, 1))
        assert_allclose(rho_A, ptrace(rho_ABC, dims, 1, 2))
        assert_allclose(rho_B, ptrace(rho_ABC, dims, 0, 2))
        assert_allclose(rho_C, ptrace(rho_ABC, dims, 0, 1))

    def test_hdet(self):
        dims = (2, 2, 2)
        s = linalg.random_quantum_state(8).reshape(dims)
        # TODO: hdet_alt seems to be broken...
        #self.assertAlmostEqual(linalg.hdet(s), linalg.hdet_alt(s))


if __name__ == '__main__':
    unittest.main()
