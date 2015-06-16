"""
XRay - Random search for inequalities in subspace.

Usage:
    xray init -i INPUT  [-o OUTPUT] [-s SUBDIM] [-l LIMIT]
    xray find -i INPUT  [-o OUTPUT] [-s SUBDIM] [-l LIMIT] -d FILE

Options:
    -i INPUT, --input INPUT         Read constraints matrix from file
    -o OUTPUT, --output OUTPUT      Write found constraints to this file
    -s SUBDIM, --subdim SUBDIM      Dimension of reduced space
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
    -d FILE, --directions FILE      Load directions from this file


Note:
    * files may be specified as '-' to use STDIN/STDOUT
    * if --subdim is omitted it defaults to the square-root of input dimension
    * the --limit constraint is needed for unbounded cones such as entropy
      cones to make sure a bounded solution exists


The outline of the algorithm is as follows:
    1. generate a random dual vector v in the subspace
    2. minimize v∙x subject to the linear system
    3. if the minimum is zero add v to output system, go to 1
    4. if H(i)=LIMIT for all i<SUBDIM v is useless, go to 1
    5. collect solution (~extremal ray) + active constraints
    6. after having done this multiple times, perform
        a) convex hull on extremal rays (+origin)
        b) FME on collected active constraints
"""

import sys
from os import path
from math import sqrt
from docopt import docopt
import numpy as np
import numpy.random
from .core.lp import Problem
from .util import (repeat, take, basis_vector, format_vector,
                   scale_to_int, VectorMemory, print_to, make_int_exact)


def orthogonal_complement(v):
    """
    Get the (orthonormal) basis vectors making up the orthogonal complement of
    the plane defined by n∙x = 0.
    """
    a = np.hstack((np.array([v]).T , np.eye(v.shape[0])))
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
    # posted by the `pystif.chull` module:
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


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'], ndmin=2)
    lp = Problem(system)
    dim = lp.num_cols

    if opts['--subdim'] is not None:
        subdim = int(opts['--subdim'])
    else:
        subdim = int(round(sqrt(dim)))

    if opts['--limit'] is not None:
        limit = float(opts['--limit'])
        for i in range(1, subdim):
            lp.set_col_bnds(i, 0, limit)

    if opts['init']:
        rays = inner_approximation(lp, subdim-1)
    elif opts['find']:
        directions = np.loadtxt(opts['--directions'], ndmin=2)[:,1:]
        rays = (find_xray(lp, v) for v in directions)

    seen = VectorMemory()

    output_file = opts['--output']
    if output_file and path.exists(output_file):
        old_findings = np.loadtxt(output_file, ndmin=2)
        for ray in old_findings:
            seen(ray)
    output = print_to(output_file, append=True)

    rays = (ray for ray in rays if not seen(ray))

    for ray in rays:
        output(format_vector(ray))


if __name__ == '__main__':
    main()
