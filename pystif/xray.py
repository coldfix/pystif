"""
XRay - Random search for inequalities in subspace.

Usage:
    xray -i INPUT  [-o OUTPUT] [-s SUBDIM] [-l LIMIT]

Options:
    -i INPUT, --input INPUT         Read constraints matrix from file
    -o OUTPUT, --output OUTPUT      Write found constraints to this file
    -s SUBDIM, --subdim SUBDIM      Dimension of reduced space
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM


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

from math import sqrt
from docopt import docopt
import numpy as np
import numpy.random
from .core.lp import Problem
from .util import repeat, basis_vector, format_vector


def random_direction_vector(dim, embed_dim):
    v = np.random.normal(size=dim)
    v /= np.linalg.norm(v)
    v.resize(embed_dim)
    return v


def main(args=None):
    opts = docopt(__doc__, args)

    system = np.loadtxt(opts['--input'])
    lp = Problem.from_matrix(system, lb_row=0)
    dim = lp.num_cols

    if opts['--subdim'] is not None:
        subdim = int(opts['--subdim'])
    else:
        subdim = int(round(sqrt(dim + 1))) - 1

    if opts['--limit'] is not None:
        limit = float(opts['--limit'])
        for i in range(subdim):
            lp.add_row(basis_vector(dim, i), ub=limit)

    directions = repeat(random_direction_vector, subdim, dim)

    for i, v in enumerate(directions):
        solution = lp.maximize(v)
        if all(solution == np.zeros(dim)):
            continue
        solution.resize(subdim)
        solution /= np.linalg.norm(solution)
        # TODO: output active constraints
        # TODO: stream this to OUTPUT
        print(format_vector(solution))
        # TODO: specify max samples on command line
        if i >= 1000000:
            return


if __name__ == '__main__':
    hull = main()
