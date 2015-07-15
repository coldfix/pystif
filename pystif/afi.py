"""
Project convex cone to subspace by an adjacent facet iteration method.

Usage:
    afi INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES]

Options:
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -f FACES, --faces FACES         File with known faces of the projected
                                    polyhedron
    -y SYM, --symmetry SYM          Symmetry group generators
"""

from functools import partial

import numpy as np
from docopt import docopt

from .core.array import scale_to_int
from .core.io import StatusInfo, System, default_column_labels, SystemFile
from .core.geom import ConvexPolyhedron
from .core.linalg import matrix_nullspace, plane_normal, addz, delz
from .core.symmetry import NoSymmetry, SymmetryGroup
from .core.util import VectorMemory
from .chm import convex_hull_method, print_status, print_qhull


def get_cut_boundaries(body, subspace):
    """
    Perform CHM on the facet to obtain a list of all its "hyper-facets".
    """
    cut = body.intersection(subspace)
    info = StatusInfo()
    callbacks = (lambda ray: None,
                 lambda facet: None,
                 partial(print_status, info),
                 partial(print_qhull, info))
    return convex_hull_method(cut, cut.basis(), *callbacks)


def adjacent_facet_iteration(polyhedron, initial_facet, found_cb, symmetries,
                             status_info):

    subdim = len(initial_facet)
    seen_b = set()
    seen = VectorMemory()

    queue = [initial_facet]
    for sym in symmetries(initial_facet):
        if not seen(initial_facet):
            found_cb(initial_facet)

    while queue:
        facet = queue.pop()
        facet = scale_to_int(facet)

        equations, subspace = get_cut_boundaries(polyhedron, facet)

        for i, equation in enumerate(delz(equations)):
            status_info(queue, equations, i)

            boundary = addz(subspace.back(matrix_nullspace([equation])))

            eq = subspace.back(equation)
            eq = np.hstack((0, eq))
            eq = scale_to_int(eq)

            adj = polyhedron.get_adjacent_facet(facet, boundary, eq)

            if not seen(adj):
                queue.append(adj)
                found_cb(adj)

                for sym in symmetries(adj):
                    if not seen(sym):
                        found_cb(sym)

        status_info(queue, equations, len(equations))


def afi_status(info, queue, equations, i):
    num_eqs = len(equations)
    len_eqs = len(str(num_eqs))
    info("FEM queue: {:4}, progress: {}/{}".format(
        len(queue),
        str(i).rjust(len_eqs),
        num_eqs,
    ))
    if i == num_eqs:
        info()


def main(args=None):
    opts = docopt(__doc__, args)

    system = System.load(opts['INPUT'])
    dim = system.dim
    if not system.columns:
        system.columns = default_column_labels(dim)

    system, subdim = system.prepare_for_projection(opts['--subspace'])
    polyhedron = ConvexPolyhedron.from_cone(system, subdim,
                                            float(opts['--limit']))

    face = np.hstack((0, np.ones(subdim-1)))
    facet = polyhedron.refine_to_facet(face)

    print("Found initial facet, starting enumeration...")

    facet_file = SystemFile(opts['--output'], columns=system.columns[:subdim])

    if opts['--symmetry']:
        col_names = system.columns[:subdim]
        symmetries = SymmetryGroup.load(opts['--symmetry'], col_names)
    else:
        symmetries = NoSymmetry

    info = StatusInfo()

    adjacent_facet_iteration(polyhedron, facet, facet_file, symmetries,
                             partial(afi_status, info))


if __name__ == '__main__':
    main()
