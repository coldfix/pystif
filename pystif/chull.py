"""
Find hull off a convex cone that may not have full dimensionality.

Usage:
    chull -i INPUT -x XRAYS [-o OUTPUT]

Options:
    -i INPUT, --input INPUT         Load initial (big) system from this file
    -x XRAYS, --xrays XRAYS         Load extremal rays from this file
    -o OUTPUT, --output OUTPUT      Save results to this file
"""

from math import log2
import numpy as np
import scipy.spatial
from docopt import docopt
from .core.lp import Problem, UnboundedError
from .core.it import elemental_inequalities
from .util import format_vector, scale_to_int


def num_vars(dim):
    return int(round(log2(dim+1)))


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
    centered = data_points - np.mean(data_points, axis=0)
    cov_mat = np.cov(centered.T)
    u, s, v = np.linalg.svd(cov_mat)
    num_comp = next((i for i, c in enumerate(s) if c < s_limit), len(s))
    return u[:,:num_comp], u[:,num_comp:]


def convex_hull(xrays):

    points = xrays

    # Now make sure the dataset lives in a full dimensional subspace
    principal_basis, subordinate_basis = principal_components(points)
    points = np.dot(points, principal_basis)

    hull = scipy.spatial.ConvexHull(points)
    equations = hull.equations
    equations = equations[:,:-1]        # TODO: last component

    # add back the removed dimensions
    equations = np.dot(equations, principal_basis.T)

    subord = subordinate_basis.T
    return np.vstack((
        subord,
        -subord,
        equations,
        -equations,
    ))


def filter_equations(big_system, equations):

    dim = big_system.shape[1]
    subdim = equations.shape[1]

    el_ineq = [ineq[1:] for ineq in elemental_inequalities(num_vars(subdim))]
    lp_origin = Problem.from_matrix(big_system, lb_row=0)
    lp_target = Problem.from_matrix(el_ineq, lb_row=0)

    for eq in equations:
        eq = scale_to_int(eq)
        eq_embedded = np.hstack((eq, np.zeros(dim-subdim)))
        if not lp_origin.has_optimal_solution(eq_embedded):
            print("# Not valid :(")
            # TODO: this case can be used to search for points
            continue
        if lp_target.has_optimal_solution(eq):
            print("# Nothing new :(")
            continue
        lp_target.add_row(eq, lb=0)
        yield eq


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'])
    dataset = np.loadtxt(opts['--xrays'])
    equations = convex_hull(dataset)

    for eq in filter_equations(system, equations):
        print(format_vector(eq))

if __name__ == '__main__':
    main()

