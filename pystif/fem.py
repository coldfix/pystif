"""
Project convex cone to subspace by an incremental facet enumeration method.

Usage:
    fem INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT]

Options:
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -f FACES, --faces FACES         File with known faces of the projected
                                    polyhedron
"""

from functools import partial
import numpy as np
import scipy
from docopt import docopt

from .core.lp import Problem
from .core.io import (StatusInfo, System, default_column_labels, SystemFile,
                      VectorMemory, scale_to_int)
from .chm import (convex_hull_method, inner_approximation, find_xray,
                  print_status, print_qhull)


matrix_rank = np.linalg.matrix_rank


def matrix_nullspace(A, eps=1e-10):
    """
    Return a basis for the solution space of ``A∙x=0`` as column vectors.
    """
    u, s, vh = np.linalg.svd(A)
    n = next((i for i, c in enumerate(s) if c < eps), len(s))
    return vh[n:]


def get_plane(v):
    """
    Get normal vector of the plane defined by the vertices (=rows of) v.
    """
    # TODO: it should be easy to obtain the result directly from the
    # facet equation, boundary equation and additional vertex without
    # resorting to matrix decomposition techniques.
    space = matrix_nullspace(v)
    assert space.shape[0] == 1
    return space[0].flatten()

    #----------------------------------------
    # TODO: The current method doesn't detect errors (e.g. if the input
    # vertices are not a body of the correct dimension). In this regard
    # principal_components should behave better.
    # v = np.atleast_2d(v)
    # a = np.hstack((v.T , np.eye(v.shape[1])))
    # q, r = np.linalg.qr(a)
    # return q[:,-1:].flatten()
    #----------------------------------------


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
    # facet.
    subdim = len(facet)
    plane = -facet

    seen = VectorMemory()

    # TODO: this algorithm walks through cycles...
    while True:
        vertex = lp.minimize(plane, embed=True)
        vertex = scale_to_int(vertex[1:subdim])
        assert not seen(vertex)
        if lp.get_objective_value() >= -atol:
            return plane
        plane = get_plane(b_simplex + (vertex,))
        plane = scale_to_int(plane)
        if np.dot(plane, old_vertex) <= -atol:
            plane = -plane
        plane = np.hstack((0, plane))


def facet_enumeration_method(lp, lpb, initial_facet, found_cb):

    subdim = len(initial_facet)

    # TODO: maintain a list[frozenset[vertices]] to never do a facet twice
    seen_b = set()
    seen = VectorMemory()

    found_cb(initial_facet)
    queue = [initial_facet]
    while queue:
        facet = queue.pop()
        facet = scale_to_int(facet)
        assert is_facet(lpb, facet)

        hull, subspace = get_facet_boundaries(lp, lpb, facet)
        # TODO: need to recover points from hull subspace
        points = [scale_to_int(np.dot(p, subspace.T)) for p in hull.points]

        for equation, simplex in zip(hull.equations, hull.simplices):
            boundary = tuple(points[i] for i in simplex)
            if not any(np.allclose(p, 0) for p in boundary):
                continue
            assert abs(equation[-1]) < 1e-10

            if matrix_rank(boundary) < subdim-3:
                # TODO: I don't know why/if it's possible for an N dimensional
                # convex polytope to have faces with less than N-2 dimensional
                # boundaries, but it apparently it happens…
                # TODO: I don't know either if it will lead to an incomplete
                # description to ignore these, but I will postpone this
                # question for later and first get it work at all.
                continue

            eq = np.dot(-equation[:-1], subspace.T)

            old_vertex = max(points, key=lambda p: np.dot(eq, p))

            adj = get_adjacent_facet(lpb, facet, boundary, old_vertex)
            # TODO: check adj against every seen facet
            if not seen(adj):
                queue.append(adj)
                found_cb(adj)


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

    facet_enumeration_method(lp, lpb, facet, facet_file)


if __name__ == '__main__':
    main()
