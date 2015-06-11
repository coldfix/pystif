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


def orthogonal_complement(v):
    """
    Get the (orthonormal) basis vectors making up the orthogonal complement of
    the plane defined by n∙x = 0.
    """
    a = np.hstack((np.array([v]).T , np.eye(v.shape[0])))
    q, r = np.linalg.qr(a)
    return q[:,1:]


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


def convex_hull(xrays):

    points = del_const_col(xrays)
    points = np.vstack((np.zeros(points.shape[1]), points))

    # Now make sure the dataset lives in a full dimensional subspace
    principal_basis, subordinate_basis = principal_components(points)
    points = np.dot(points, principal_basis)

    hull = scipy.spatial.ConvexHull(points)
    faces = hull.equations

    faces = np.array([
        eq[:-1] for eq in faces
        if abs(eq[-1]) < 1e-5])

    # add back the removed dimensions
    faces = np.dot(faces, principal_basis.T)

    nullspace = subordinate_basis.T
    nullspace = add_left_zero(nullspace)
    faces = add_left_zero(faces)
    return np.vstack((
        nullspace,
        -nullspace,
        # The following is an empirical minus sign. I didn't find anything on
        # the qhull documentation as to how the equations are oriented, but
        # this seems to work. As soon as things start failing, you might want
        # to take a second look.
        -faces,
    ))


def unique_rows(L):
    """Return new matrix with the unique rows of L."""
    seen = VectorMemory()
    L = list(map(scale_to_int, L))
    return np.array([q for q in L if not seen(q)])


def classify(lp, ineqs, report_yes, report_no, status_info):
    """
    Classify inequalities according to whether they are implied by the LP.

    All inequalities will be fed to either ``report_yes`` or ``report_no``
    depending on whether they are implied.

    Returns ``True`` if all inequalities are valid.
    """
    total = len(ineqs)
    yes = 0
    for i, q in enumerate(ineqs):
        status_info(i, total, yes)
        if lp.implies(q, embed=True):
            yes += 1
            report_yes(q)
        else:
            report_no(q)
    status_info(total, total, yes)
    return yes == total


def print_vector(print_, q):
    """Print formatted vector q."""
    print_(format_vector(q))


def print_status(print_, i, total, yes):
    """Print status."""
    l = len(str(total))
    print_("Progress: {}/{},  valid: {}"
           .format(str(i).rjust(l), total, str(yes).rjust(l)))
    if i == total:
        print_("\n")


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'], ndmin=2)
    xrays = np.loadtxt(opts['--xrays'], ndmin=2)
    lp = Problem(system)

    faces = convex_hull(xrays)
    faces = unique_rows(faces)

    feedback = print_to(opts['--feedback'], '#', default=sys.stdout)
    output = print_to(opts['--output'])
    info = partial(print, '\r', end='', file=sys.stderr)

    callbacks = (partial(print_vector, output),
                 partial(print_vector, feedback),
                 partial(print_status, info))
    return classify(lp, faces, *callbacks)


if __name__ == '__main__':
    sys.exit(0 if main() else 17)
