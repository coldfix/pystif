"""
Find projection of a convex cone to a lower dimensional subspace.

Usage:
    chm INPUT -s SUBSPACE [-o OUTPUT] [-x XRAYS] [-l LIMIT] [-r] [-q] [-i FILE]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -x XRAYS, --xrays XRAYS         Save projected extremal rays to this file
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -r, --resume                    Resume using previously computed rays
                                    (must be fully dimensional!)
    -q, --quiet                     Less status output
    -i FILE, --info FILE            Print short summary to file (YAML)

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

from functools import partial

import numpy as np
import scipy.spatial

from .core.app import application
from .core.array import scale_to_int
from .core.io import SystemFile
from .core.geom import LinearSubspace
from .core.util import VectorMemory


def convex_hull_method(polyhedron, rays,
                       report_ray, report_yes,
                       status_info, qinfo):

    result = []

    points = np.copy(rays)

    # Now make sure the dataset lives in a full dimensional subspace
    subspace = LinearSubspace.from_rowspace(points)

    for face in polyhedron.nullspace_int():
        report_yes(face)
        report_yes(-face)

    # initial hull
    points = subspace.into(points)
    points = np.vstack((np.zeros(subspace.dim), points))
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
            face = subspace.back(face)
            face = scale_to_int(face)

            if seen(face):
                continue

            if polyhedron.is_face(face):
                yes += 1
                report_yes(face)
                result.append(face)
            else:
                ray = polyhedron.search(face)
                if seen_ray(ray):
                    continue
                report_ray(ray)
                new_points.append(subspace.into(ray))

        if new_points:
            status_info(total, total, yes)
            points = np.vstack((points, new_points))
            qinfo(len(points))
            hull.add_points(new_points, restart=True)
        else:
            break

    status_info(total, total, yes)
    return result, subspace


def print_status(print_, i, total, yes):
    """Print status."""
    l = len(str(total))
    print_("CHM Progress: {}/{} ({} facets)"
           .format(str(i).rjust(l), total, yes))
    if i == total:
        print_()


def print_qhull(print_, num_points):
    print_("  > qhull on {} rays\n".format(num_points))


@application
def main(app):
    ray_file = SystemFile(app.opts['--xrays'], append=app.resume, default=None,
                          columns=app.system.columns[:app.subdim])

    if ray_file._matrix:
        rays = ray_file._matrix
    else:
        rays = app.polyhedron.basis()
        for ray in rays:
            ray_file(ray)
        ray_file._print()

    info = app.info(0)
    callbacks = (ray_file,
                 app.output,
                 partial(print_status, info),
                 partial(print_qhull, info))

    app.start_timer()
    convex_hull_method(app.polyhedron, rays, *callbacks)
