"""
Points to facets utility.

Usage:
    p2f INPUT -P POINTS -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-q]... [-p]

Options:
    -P POINTS, --points POINTS      File with points
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or list)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -y SYM, --symmetry SYM          Symmetry group generators
    -q, --quiet                     Show less output
    -p, --pretty                    Pretty print output inequalities

"""

import numpy as np

from .core.app import application
from .core.lp import Problem
from .core.io import System
from .core.util import scale_to_int


def outer_points_to_facets(polyhedron, system, points):

    """
    Get facets that show that the given points are not in the interior of the
    polytope.

    All points must be either on the hull or outside of the polytope.
    """

    L = system.matrix.T
    _, num_cols = L.shape
    dim = polyhedron.dim

    qlp = Problem(                      # Find f = qL s.t.
        num_cols=num_cols,              #
        lb_col=0)                       #    q_i ≥ 0  ∀ i
    qlp.add(np.ones(num_cols), 1, 1)    #   Σq_i = 1
    qlp.add(L[dim:], 0, 0)              # (qL)_i = 0  ∀ i > m

    for x in points:
        q = qlp.minimize(x @ L[:dim])   # min qLx
        q = scale_to_int(q)
        f = (L @ q)[:dim]               # f = qL
        f = scale_to_int(f)
        yield from polyhedron.face_to_facets(f)


@application
def main(app):
    points = System.load(app.opts['--points'])
    points, _ = points.slice(app.system.columns[:app.subdim])
    for f in outer_points_to_facets(app.polyhedron, app.system, points.matrix):
        app.report_facet(f)
