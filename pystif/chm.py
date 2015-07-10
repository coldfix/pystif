"""
Find projection of a convex cone to a lower dimensional subspace.

Usage:
    chm INPUT -s SUBSPACE [-o OUTPUT] [-x XRAYS] [-l LIMIT] [-r]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -x XRAYS, --xrays XRAYS         Save projected extremal rays to this file
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -r, --resume                    Resume using previously computed rays
                                    (must be fully dimensional!)

Note:
    * output files may be specified as '-' to use STDIN/STDOUT
    * --subspace can either be the name of a file containing the column names
      of the subspace or the number of leftmost columns
    * the --limit constraint is needed for unbounded cones such as entropy
      cones to make sure a bounded solution exists

The outline of the algorithm is as follows:
    1. generate a set of extremal rays that span the subspace in which the
       projected cone lives
    2. compute the convex hull over the current set of extremal rays
    3. for each facet decide whether it is also a facet of the actual LP
        a) yes: output facet normal vector
        b) no: find the extremal ray that is outside the facet, go to 2.
"""

import sys
from functools import partial
import numpy as np
import numpy.random
import scipy.spatial
from docopt import docopt
from .core.io import (scale_to_int, make_int_exact, VectorMemory, System,
                      default_column_labels, SystemFile, StatusInfo)


def orthogonal_complement(v):
    """
    Get the (orthonormal) basis vectors making up the orthogonal complement of
    the plane defined by n∙x = 0.
    """
    v = np.atleast_2d(v)
    a = np.hstack((v.T , np.eye(v.shape[1])))
    q, r = np.linalg.qr(a)
    return q[:,1:]


def random_direction_vector(dim):
    v = np.random.normal(size=dim)
    v /= np.linalg.norm(v)
    return v


def find_xray(lp, direction):
    subdim = 1+len(direction)
    direction = np.hstack((0, direction, np.zeros(lp.num_cols-subdim)))
    # For the random vectors it doesn't matter whether we use `minimize` or
    # `maximize` — but it *does* matter for the oriented direction vectors
    # obtained from other functions:
    xray = lp.minimize(direction)
    xray.resize(subdim)
    # xray[0] is always -1, this prevents any shortening if we decide to that
    # later on, so just set it to 0.
    xray[0] = 0
    xray = scale_to_int(xray)
    # TODO: output active constraints
    return xray


def inner_approximation(lp, dim):
    """
    Return a matrix of extreme points whose convex hull defines an inner
    approximation to the projection of the polytope defined by the LP to the
    subspace of its lowest ``dim`` components.

    Extreme points are returned as matrix rows.

    The returned points define a polytope with the dimension of the projected
    polytope itself (which may be less than ``dim``).
    """
    # FIXME: Better use deterministic or random direction vectors?
    v = random_direction_vector(dim)
    points = np.array([
        find_xray(lp, v)[1:],
    ])
    orth = np.eye(dim)
    while orth.shape[1] > 0:
        # Generate arbitrary vector in orthogonal space and min/max along its
        # direction:
        d = random_direction_vector(orth.shape[1])
        v = np.dot(d, orth.T)
        x = find_xray(lp, v)[1:]
        p = make_int_exact(np.dot(x-points[0], orth))
        if all(p == 0):
            x = find_xray(lp, -v)[1:]
            p = make_int_exact(np.dot(x-points[0], orth))
        if all(p == 0):
            # Optimizing along ``v`` yields a vector in our ray space. This
            # means ``v∙x=0`` is part of the LP.
            orth = np.dot(orth, orthogonal_complement(d))
        else:
            # Remove discovered ray from the orthogonal space:
            orth = np.dot(orth, orthogonal_complement(p))
            points = np.vstack((points, x))
    return np.hstack((np.zeros((points.shape[0], 1)), points))


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
    return hull, subspace


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

    resume = opts['--resume']
    facet_file = SystemFile(opts['--output'], append=resume,
                            columns=system.columns[:subdim])
    ray_file = SystemFile(opts['--xrays'], append=resume, default=None,
                          columns=system.columns[:subdim])

    if ray_file._matrix:
        rays = ray_file._matrix
    else:
        rays = inner_approximation(lpb, subdim-1)
        for ray in rays:
            ray_file(ray)
        ray_file._print()

    info = StatusInfo()

    callbacks = (ray_file,
                 facet_file,
                 partial(print_status, info),
                 partial(print_qhull, info))

    convex_hull_method(lp, lpb, rays, *callbacks)


if __name__ == '__main__':
    main()
