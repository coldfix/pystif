"""
Find hull off a convex cone that may not have full dimensionality.

Usage:
    chull -i INPUT -x XRAYS [-o OUTPUT] [-f FEEDBACK]

Options:
    -i INPUT, --input INPUT         Load initial (big) system from this file
    -x XRAYS, --xrays XRAYS         Load extremal rays from this file
    -o OUTPUT, --output OUTPUT      Save results to this file
    -f FILE, --feedback FILE        Save pending normal vectors to file

Exit codes:

    - 0 — full projection to the subspace has been computed
    - 17 — still missing extremal rays for complete subspace description
    - other exit codes correspond to program errors
"""

import sys
from functools import partial
import numpy as np
import scipy.spatial
from docopt import docopt
from .core.lp import Problem
from .core.it import elemental_inequalities, num_vars
from .util import format_vector, scale_to_int, VectorMemory, print_to
from .xray import find_xray


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
    xrays = np.loadtxt(opts['--xrays'], ndmin=2)
    lp = Problem(system)

    lpb = Problem(system)

    subdim = xrays.shape[1]
    limit = 1
    for i in range(1, subdim):
        lpb.set_col_bnds(i, 0, limit)



    feedback = print_to(opts['--feedback'], '#', default=sys.stdout)
    output = print_to(opts['--output'])

    info = partial(print, '\r', end='', file=sys.stderr)

    callbacks = (partial(print_vector, feedback),
                 partial(print_vector, output),
                 partial(print_status, info),
                 partial(print_qhull, info))

    convex_hull_method(lp, lpb, xrays, *callbacks)
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 17)
