"""
Find hull off a convex cone that may not have full dimensionality.

Usage:
    chull -i INPUT -x XRAYS [-o OUTPUT] [-f FEEDBACK] [-e]

Options:
    -i INPUT, --input INPUT         Load initial (big) system from this file
    -x XRAYS, --xrays XRAYS         Load extremal rays from this file
    -o OUTPUT, --output OUTPUT      Save results to this file
    -f FILE, --feedback FILE        Save pending normal vectors to file
    -e, --el-ineqs                  Don't show elemental inequalities
"""

import sys
from functools import partial
import numpy as np
import scipy.spatial
from docopt import docopt
from .core.lp import Problem, UnboundedError
from .core.it import elemental_inequalities, num_vars
from .util import format_vector, scale_to_int, VectorMemory


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
    equations = hull.equations

    equations = np.array([
        eq[:-1] for eq in equations
        if abs(eq[-1]) < 1e-5])

    # add back the removed dimensions
    equations = np.dot(equations, principal_basis.T)

    subord = subordinate_basis.T
    subord = add_left_zero(subord)
    equations = add_left_zero(equations)
    return np.vstack((
        subord,
        -subord,
        -equations,
        equations,
    ))


def info(*args, file=sys.stderr):
    print('\r', *args, end='', file=file)


def filter_equations(big_system, equations, feedback, el_ineqs):

    dim = big_system.shape[1]
    subdim = equations.shape[1]

    lp_origin = Problem(big_system)
    lp_target = Problem(num_cols=subdim)
    if el_ineqs:
        lp_target.add(list(elemental_inequalities(num_vars(subdim))))

    seen = VectorMemory()

    num_unseen = 0
    num_feedback = 0
    num_trivial = 0
    for i, eq in enumerate(equations):
        info("Progress: {:5}/{} | {:4} {:4}/{:4}"
             .format(i, len(equations),
                     num_trivial, num_feedback, num_unseen))
        eq = scale_to_int(eq)
        if seen(eq):
            continue
        num_unseen += 1
        eq_embedded = np.hstack((eq, np.zeros(dim-subdim)))
        if not lp_origin.has_optimal_solution(eq_embedded):
            num_feedback += 1
            feedback(format_vector(eq))
            continue
        if lp_target.has_optimal_solution(eq):
            num_trivial += 1
            continue
        lp_target.add_row(eq, lb=0)
        yield eq


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'])
    dataset = np.loadtxt(opts['--xrays'])
    equations = convex_hull(dataset)

    if opts['--feedback']:
        feedback_file = open(opts['--feedback'], 'w', buffering=1)
        feedback = partial(print, file=feedback_file)
    else:
        feedback = partial(print, '#', file=sys.stdout)

    el_ineqs = opts['--el-ineqs']

    for eq in filter_equations(system, equations, feedback, el_ineqs):
        print(format_vector(eq))
    print("", file=sys.stderr)


if __name__ == '__main__':
    main()
