"""
Points to facets utility.

Usage:
    p2f INPUT -p POINTS -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-q]...

Options:
    -p POINTS, --points POINTS      File with points
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -y SYM, --symmetry SYM          Symmetry group generators
    -q, --quiet                     Show less output

"""

import numpy as np

from .core.app import application
from .core.lp import Problem
from .core.linalg import as_column_vector
from .core.io import System
from .core.util import scale_to_int


def p2f(polyhedron, system, points):

    L = system.matrix
    num_rows, num_cols = L.shape
    dim = polyhedron.dim

    qlp = Problem(                      # Find f = qL s.t.
        num_cols=num_rows,              #
        lb_col=0)                       #    q_i ≥ 0  ∀ i
    qlp.add(np.ones(num_rows), 1, 1)    #   Σq_i = 1
    qlp.add(L.T[dim:], 0, 0)            # (qL)_i = 0  ∀ i > m

    for x in points:

        q = qlp.minimize(x @ L.T[:dim])
        q = scale_to_int(q)

        f = (L.T @ q)[:dim]
        f = scale_to_int(f)

        yield polyhedron.refine_to_facet(f)


@application
def main(app):

    points = System.load(app.opts['--points'])
    points, _ = points.slice(app.system.columns[:app.subdim])

    facets = p2f(app.polyhedron, app.system, points.matrix)

    for f in facets:
        for g in app.symmetries(f):
            app.output(g)
