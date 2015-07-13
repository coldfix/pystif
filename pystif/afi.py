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
from .core.linalg import matrix_nullspace, plane_normal
from .core.symmetry import NoSymmetry, SymmetryGroup
from .core.util import VectorMemory
from .chm import (convex_hull_method, inner_approximation,
                  print_status, print_qhull)


def get_facet_boundaries(lp, lpb, facet):
    """
    Perform CHM on the facet to obtain a list of all its "hyper-facets".
    """
    lp = lp.copy()
    lpb = lpb.copy()
    lp.add_row(facet, 0, 0, embed=True)
    lpb.add_row(facet, 0, 0, embed=True)
    subdim = len(facet)
    rays = inner_approximation(lpb, subdim-1)
    info = StatusInfo()
    callbacks = (lambda ray: None,
                 lambda facet: None,
                 partial(print_status, info),
                 partial(print_qhull, info))
    return convex_hull_method(lp, lpb, rays, *callbacks)


def get_adjacent_facet(lp, facet, b_simplex, old_vertex, atol=1e-10):
    """
    Get the adjacent facet defined by `facet` and its boundary
    `facet_boundary`.
    """
    # TODO: in first step maximize along plane defined by facet_boundary and
    # facet?
    subdim = len(facet)
    plane = -facet
    seen = VectorMemory()
    while True:
        vertex = lp.minimize(plane, embed=True)
        vertex = scale_to_int(vertex[1:subdim])
        assert not seen(vertex)
        if lp.get_objective_value() >= -atol:
            return plane
        # TODO: it should be easy to obtain the result directly from the
        # facet equation, boundary equation and additional vertex without
        # resorting to matrix decomposition techniques.
        plane = plane_normal(b_simplex + (vertex,))
        plane = scale_to_int(plane)
        if np.dot(plane, old_vertex) <= -atol:
            plane = -plane
        plane = np.hstack((0, plane))


def adjacent_facet_iteration(lp, lpb, initial_facet, found_cb, symmetries):

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
        assert is_facet(lpb, facet)

        equations, subspace = get_facet_boundaries(lp, lpb, facet)

        # for status output:
        num_eqs = len(equations)
        len_eqs = len(str(num_eqs))

        for i, equation in enumerate(equations):
            boundary = tuple(np.dot(matrix_nullspace([equation]), subspace.T))

            print("\rFEM queue: {:4}, progress: {}/{}".format(
                len(queue),
                str(i).rjust(len_eqs),
                num_eqs,
            ), end='')

            eq = np.dot(equation, subspace.T)

            adj = get_adjacent_facet(lpb, facet, boundary, eq)

            if not seen(adj):
                queue.append(adj)
                found_cb(adj)

                for sym in symmetries(adj):
                    if not seen(sym):
                        found_cb(sym)


def filter_non_singular_directions(lp, nullspace):
    for i, direction in enumerate(nullspace):
        if lp.implies(np.hstack((0, direction)), embed=True):
            direction = -direction
            if lp.implies(np.hstack((0, direction)), embed=True):
                continue
        yield i, direction


def refine_to_facet(lp, face):
    subspace = intersect_polyhedron_with_face(lp, face)[:,1:]
    nullspace = matrix_nullspace(np.vstack((subspace, face[1:])))
    try:
        i, direction = next(filter_non_singular_directions(lp, nullspace))
    except StopIteration:
        return face
    simplex = tuple(subspace) + tuple(np.delete(nullspace, i, axis=0))
    plane = get_adjacent_facet(lp, face, simplex, direction)
    return refine_to_facet(lp, plane)


def intersect_polyhedron_with_face(lp, face):
    subdim = len(face)
    lp = lp.copy()
    lp.add(face, 0, 0, embed=True)
    return inner_approximation(lp, subdim-1)


def is_facet(lp, plane):
    subdim = len(plane)
    body_dim = len(inner_approximation(lp, subdim-1))
    face_dim = len(intersect_polyhedron_with_face(lp, plane))
    return face_dim+1 == body_dim


def main(args=None):
    opts = docopt(__doc__, args)

    system = System.load(opts['INPUT'])
    dim = system.dim
    if not system.columns:
        system.columns = default_column_labels(dim)

    system, subdim = system.prepare_for_projection(opts['--subspace'])

    lp = system.lp()
    lpb = system.lp()
    if opts['--limit'] is not None:
        limit = float(opts['--limit'])
        for i in range(1, subdim):
            lpb.set_col_bnds(i, 0, limit)

    face = np.ones(subdim)
    face[0] = 0
    facet = refine_to_facet(lpb, face)

    assert lp.implies(facet, embed=True)
    assert is_facet(lpb, facet)

    print("Found initial facet, starting enumeration...")

    facet_file = SystemFile(opts['--output'], columns=system.columns[:subdim])

    if opts['--symmetry']:
        col_names = system.columns[:subdim]
        symmetries = SymmetryGroup.load(opts['--symmetry'], col_names)

    else:
        symmetries = NoSymmetry

    adjacent_facet_iteration(lp, lpb, facet, facet_file, symmetries)


if __name__ == '__main__':
    main()
