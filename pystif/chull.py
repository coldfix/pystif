"""
Find projection of a convex cone to a lower dimensional subspace.

Usage:
    chull -i INPUT [-o OUTPUT] [-x XRAYS] [-s SUBDIM] [-l LIMIT] [-r]

Options:
    -i INPUT, --input INPUT         Load LP from this file
    -o OUTPUT, --output OUTPUT      Save valid facets to this file
    -x XRAYS, --xrays XRAYS         Save extremal rays to this file
    -s SUBDIM, --subdim SUBDIM      Dimension of reduced space
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
    -r, --resume                    Resume with previously computed rays
"""

import os
import sys
from math import sqrt
from functools import partial
import numpy as np
import scipy.spatial
from docopt import docopt
from .core.lp import Problem
from .core.it import elemental_inequalities, num_vars
from .util import format_vector, scale_to_int, VectorMemory, print_to
from .xray import find_xray, inner_approximation


def principal_components(data_points, s_limit=1e-10):
    """
    Get the (orthonormal) basis vectors of the principal components of the
    data set specified by the rows of M.
    """
    cov_mat = np.cov(data_points.T)
    u, s, v = np.linalg.svd(cov_mat)
    num_comp = next((i for i, c in enumerate(s) if c < s_limit), len(s))
    return u[:,:num_comp], u[:,num_comp:]


def del_const_col(mat):
    return mat[:,1:]


def add_left_zero(mat):
    return np.hstack((np.zeros((mat.shape[0], 1)), mat))


def convex_hull_method(lp, lpb, rays,
                       report_ray, report_yes,
                       status_info, qinfo):

    points = del_const_col(rays)
    points = np.vstack((np.zeros(points.shape[1]), points))

    # Now make sure the dataset lives in a full dimensional subspace
    subspace, nullspace = principal_components(points)

    nullspace = nullspace.T
    nullspace = add_left_zero(nullspace)
    for face in nullspace:
        report_yes(face)
        report_yes(-face)

    if nullspace.shape[0] == 0:
        subspace = np.eye(points.shape[1])

    # initial hull
    points = np.dot(points, subspace)
    qinfo(len(points))
    hull = scipy.spatial.ConvexHull(points, incremental=True)

    yes = 0
    seen = VectorMemory()
    seen_ray = VectorMemory()
    for ray in rays:
        seen_ray(ray)

    while True:
        new_points = []
        faces = hull.equations
        total = faces.shape[0]
        for i, face in enumerate(faces):
            if abs(face[-1]) > 1e-5:
                continue
            status_info(i, total, yes)

            # The following is an empirical minus sign. I didn't find anything
            # on the qhull documentation as to how the equations are oriented,
            # but apparently points x inside the convex hull are described by
            # ``face ∙ (x,1) ≤ 0``
            face = -face[:-1]
            face = np.dot(face, subspace.T)
            face = np.hstack((0, face))
            face = scale_to_int(face)

            if seen(face):
                continue

            if lp.implies(face, embed=True):
                yes += 1
                report_yes(face)
            else:
                ray = find_xray(lpb, face[1:])
                if seen_ray(ray):
                    continue
                report_ray(ray)
                point = np.dot(ray[1:], subspace)
                new_points.append(point)

        if new_points:
            status_info(total, total, yes)
            points = np.vstack((points, new_points))
            qinfo(len(points))
            hull.add_points(new_points, restart=True)
        else:
            break

    status_info(total, total, yes)


def print_vector(print_, q):
    """Print formatted vector q."""
    print_(format_vector(q))


def print_status(print_, i, total, yes):
    """Print status."""
    l = len(str(total))
    print_("Progress: {}/{} ({} facets)"
           .format(str(i).rjust(l), total, yes))
    if i == total:
        print_("\n")


def print_qhull(print_, num_points):
    print_("qhull on {} rays\n".format(num_points))


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'], ndmin=2)
    lp = Problem(system)
    dim = lp.num_cols

    if opts['--subdim'] is not None:
        subdim = int(opts['--subdim'])
    else:
        subdim = int(round(sqrt(dim)))

    lpb = Problem(system)
    if opts['--limit'] is not None:
        limit = float(opts['--limit'])
        for i in range(1, subdim):
            lpb.set_col_bnds(i, 0, limit)

    resume = opts['--resume']
    rays_file = opts['--xrays']
    if rays_file and resume and os.path.exists(rays_file):
        xrays = np.loadtxt(rays_file, ndmin=2)
    else:
        xrays = inner_approximation(lpb, subdim-1)

    devnull = open(os.devnull, 'w')
    xrays_sink = print_to(opts['--xrays'], append=resume, default=devnull)
    facet_sink = print_to(opts['--output'], append=resume)

    info = partial(print, '\r', end='', file=sys.stderr)

    callbacks = (partial(print_vector, xrays_sink),
                 partial(print_vector, facet_sink),
                 partial(print_status, info),
                 partial(print_qhull, info))

    convex_hull_method(lp, lpb, xrays, *callbacks)


if __name__ == '__main__':
    main()
