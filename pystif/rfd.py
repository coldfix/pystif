"""
Randomized facet discovery - search for random facets of the projection of a
polyhedron.

Usage:
    rfd INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-r NUM] [-q]... [-d DIM]

Options:
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -f FACES, --faces FACES         File with known faces of the projected
                                    polyhedron
    -y SYM, --symmetry SYM          Symmetry group generators
    -r NUM, --runs NUM              Number of runs [default: 100]
    -q, --quiet                     Show less output
    -d DIM, --slice-dim DIM         Sub-slice dimension [default: 0]
"""

from functools import partial

import numpy as np

from .core.app import application
from .core.io import StatusInfo
from .core.geom import (ConvexPolyhedron, LinearSubspace,
                        random_direction_vector)
from .core.linalg import addz
from .core.util import VectorMemory


def rfd(polyhedron, symmetries, found_cb, runs, status):

    face = np.hstack((0, np.ones(polyhedron.dim-1)))
    seen = VectorMemory()

    for i in range(runs):
        status(i, runs, seen)
        facet = polyhedron.refine_to_facet(face)
        if seen(facet):
            continue
        for f in symmetries(facet):
            found_cb(f)
            seen(f)


def random_subspace(sub_dim, tot_dim):
    orth = LinearSubspace.all_space(tot_dim)
    while (tot_dim - orth.dim) < sub_dim:
        vec = orth.back(random_direction_vector(orth.dim))
        orth = LinearSubspace.from_nullspace(np.vstack((
            orth.normals,
            vec,
        )))
    return orth.nullspace()


def rss(system, polyhedron, symmetries, found_cb, runs, slice_dim, status, sub_info):

    import random
    from .chm import convex_hull_method, print_status, print_qhull
    from .core.linalg import basis_vector

    subdim = polyhedron.dim

    face = np.hstack((0, np.ones(polyhedron.dim-1)))
    seen = VectorMemory()

    for i in range(runs):

        status(i, runs, seen)
        cols = sorted(random.sample(range(1, subdim), slice_dim))
        sys2, _ = system.prepare_for_projection(cols)
        poly2 = ConvexPolyhedron.from_cone(sys2, slice_dim+1, limit=1)

        callbacks = (lambda ray: None,
                     lambda facet: None,
                     partial(print_status, sub_info),
                     partial(print_qhull, sub_info))
        faces, _ = convex_hull_method(poly2, poly2.basis(), *callbacks)

        subb = np.array([
            basis_vector(subdim, c)
            for c in [0] + cols
        ])

        for face in faces:
            status(i, runs, seen)
            face = face @ subb
            facet = polyhedron.refine_to_facet(face)
            if seen(facet):
                continue
            for f in symmetries(facet):
                found_cb(f)
                seen(f)


def rfd_status(info, i, total, seen):
    info("RFD iteration {}/{}, discovered facets: {}".format(
        str(i).rjust(len(str(total))),
        total,
        len(seen),
    ))


@application
def main(app):
    runs = int(app.opts['--runs'])
    slice_dim = int(app.opts['--slice-dim'])
    status = partial(rfd_status, app.info(1))

    app.report_nullspace()

    if slice_dim:
        rss(app.system, app.polyhedron, app.symmetries, app.output, runs, slice_dim, status,
            app.info(0))
    else:
        rfd(app.polyhedron, app.symmetries, app.output, runs, status)
