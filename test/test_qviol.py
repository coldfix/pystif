
import unittest

from itertools import product

import numpy as np

import pystif.qviol
from pystif.core.symmetry import SymmetryGroup


class CGLMP2_specialized(pystif.qviol.CGLMP2):

    """
    Check the CGLMP constraint for 3 outcomes `I₃ ≤ 2` which is given by:

        I3 = (P(A1=B1) + P(B1=A2+1) + P(A2=B2) + P(B2=A1))
             - (P(A1=B1-1) + P(B1=A2) + P(A2=B2-1) + P(B2=A1-1))
    """

    cols = ('_AB', '_Ab', '_aB', '_ab',
            '_AC', '_Ac', '_aC', '_ac',
            '_BC', '_Bc', '_bC', '_bc',)

    coef = {
        '_AB': [(1, -1, 0), (0, 1, -1), (-1, 0, 1)],    # P(B1=A1) - P(B1=A1+1)
        '_Ab': [(1, 0, -1), (-1, 1, 0), (0, -1, 1)],    # P(B2=A1) - P(B2=A1-1)
        '_aB': [(-1, 1, 0), (0, -1, 1), (1, 0, -1)],    # P(B1=A2+1) - P(B1=A2)
        '_ab': [(1, -1, 0), (0, 1, -1), (-1, 0, 1)],    # P(B2=A2) - P(B2=A2+1)
    }

    def __init__(self, system):
        self.system = system

        zero = np.zeros((3, 3))
        expr = np.array([self.coef.get(c, zero) for c in self.cols])

        # translate every party into 3 measurements each
        repl = {'A': 'IJK', 'B': 'RST', 'C': 'XYZ',
                'a': 'ijk', 'b': 'rst', 'c': 'xyz'}

        spec = (('IJK',    'ijk'),
                ('IJKijk', 'RSTrst'),
                ('IJKijk', 'XYZxyz'))
        cols = ['_' + x + y
                for ab in self.cols
                for x, y in product(*(repl[v] for v in ab[1:]))]
        sg = SymmetryGroup.load(spec, cols)

        self.matrix = list(sg(np.array(expr.flat)))


class TestQviol(unittest.TestCase):

    def test_CGLMP_constraints_generation(self):

        I3_explicit = CGLMP2_specialized(None).matrix
        I3_generic = pystif.qviol.CGLMP2(None).matrix

        self.maxDiff = None
        self.assertCountEqual(
            list(tuple(r) for r in I3_explicit),
            list(tuple(r) for r in I3_generic),
        )


if __name__ == '__main__':
    unittest.main()
