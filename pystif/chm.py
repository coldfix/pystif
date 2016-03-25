"""
Find projection of a convex cone to a lower dimensional subspace.

Usage:
    chm INPUT -s SUBSPACE [-o OUTPUT] [-x XRAYS] [-l LIMIT] [-r] [-q]... [-v]... [-p] [-i FILE]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -x XRAYS, --xrays XRAYS         Save projected extremal rays to this file
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -r, --resume                    Resume using previously computed rays
                                    (must be fully dimensional!)
    -q, --quiet                     Less status output
    -v, --verbose                   Show more output
    -p, --pretty                    Pretty print output inequalities
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
import random

import numpy as np
import scipy.spatial

from .core.app import application
from .core.array import scale_to_int
from .core.io import SystemFile
from .core.geom import LinearSubspace
from .core.util import VectorMemory


def ConvexHull(points, *, retries=5):
    """
    Wrapper for scipy.spatial.ConvexHull. The wrapper catches the occasional
    QhullError and tries to recover multiple times.
    """
    points = list(points)
    # Compute the convex hull. If it fails, try to recover a few times by
    # mixing up the input:
    for i in range(retries):
        try:
            return scipy.spatial.ConvexHull(points)
        except scipy.spatial.qhull.QhullError:
            random.shuffle(points)
    # one last time - but this time let exceptions through:
    return scipy.spatial.ConvexHull(points)


def NoSymmetry(v):
    return [v]


class CHM:

    """
    Driver for the CHM algorithm.
    """

    def __init__(self, rays, qinfo, *, retries=5, symmetries=None):
        self.qinfo = qinfo
        self.retries = retries
        self.symmetries = symmetries or NoSymmetry
        # Setup cache to avoid multiple computation:
        self.seen_face = VectorMemory()
        self.seen_ray = VectorMemory()
        # Make sure the dataset lives in a full dimensional subspace
        self.subspace = LinearSubspace.from_rowspace(rays)
        self.all_rays = []
        self.new_rays = [np.zeros(self.subspace.dim)]
        for r in rays:
            self.add(r)

    def add(self, ray):
        """Add another ray to the list of extreme rays."""
        if ray in self.seen_ray:
            return False
        for r in self.symmetries(ray):
            if self.subspace.contains(r) and not self.seen_ray(r):
                self.new_rays.append(self.subspace.into(r))
        return True

    def compute(self):
        """Compute the convex hull with the newly added points."""
        # Prepend new rays, such that the set of the first few items is always
        # different. This may be important to avoid running repeatedly into
        # the same QhullError while retaining the list order:
        self.all_rays = self.new_rays + self.all_rays
        self.new_rays = []
        self.qinfo(len(self.all_rays), self.subspace.dim)
        return ConvexHull(self.all_rays, retries=self.retries)

    def filter(self, qhull_equations):
        for face in qhull_equations:
            # Filter non-conal faces
            if abs(face[-1]) > 1e-5:
                continue
            # The following is an empirical minus sign. I didn't find anything
            # on the qhull documentation as to how the equations are oriented,
            # but apparently points x inside the convex hull are described by
            # ``face ∙ (x,1) ≤ 0``
            face = -face[:-1]
            face = self.subspace.back(face)
            face = scale_to_int(face)
            if not self.seen_face(face):
                yield face


def convex_hull_method(polyhedron, rays,
                       report_ray, report_yes,
                       status_info, qinfo,
                       *, symmetries=None):

    # List of (non-trivial) facets
    result = []

    # Report the trivial facets:
    for face in polyhedron.nullspace_int():
        report_yes(face)
        report_yes(-face)

    chm = CHM(rays, qinfo, symmetries=symmetries)

    while chm.new_rays:
        hull = chm.compute()
        total = len(hull.equations)
        status_info(0, total, len(result))

        for i, face in enumerate(chm.filter(hull.equations)):
            status_info(i, total, len(result))

            if polyhedron.is_face(face):
                # found facet:
                report_yes(face)
                result.append(face)
            else:
                # not valid - search a violating extreme point:
                ray = polyhedron.search(face)
                if chm.add(ray):
                    report_ray(ray)

        status_info(total, total, len(result))

    return result, chm.subspace


def print_status(print_, i, total, yes):
    """Print status."""
    l = len(str(total))
    print_("  -> checking hull {}/{} ({} facets)"
           .format(str(i).rjust(l), total, yes))
    if i == total:
        print_()


def print_qhull(print_, num_points, dim):
    print_("CHM: computing {}D hull of {} rays…\n".format(dim, num_points))


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
    convex_hull_method(app.polyhedron, rays, *callbacks,
                       symmetries=app.symmetries)
