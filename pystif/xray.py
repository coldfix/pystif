"""
XRay - Random search for inequalities in subspace.

Usage:
    xray -i INPUT  [-o OUTPUT] [-s SUBDIM] [-l LIMIT] [-n NUM | -d FILE]

Options:
    -i INPUT, --input INPUT         Read constraints matrix from file
    -o OUTPUT, --output OUTPUT      Write found constraints to this file
    -s SUBDIM, --subdim SUBDIM      Dimension of reduced space
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
    -n NUM, --num-samples NUM       Number of trials [default: 100000]
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
                   scale_to_int, VectorMemory)


def random_direction_vector(dim, embed_dim):
    v = np.random.normal(size=dim)
    v /= np.linalg.norm(v)
    return v


def find_xray(lp, direction):
    subdim = len(direction)
    direction = np.hstack((direction, np.zeros(lp.num_cols-subdim)))
    xray = lp.maximize(direction)
    xray.resize(subdim)
    xray = scale_to_int(xray)
    # TODO: output active constraints
    return xray


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

    if opts['--directions']:
        directions = np.loadtxt(opts['--directions'])
    else:
        num_samples = int(opts['--num-samples'])
        directions = repeat(random_direction_vector, subdim, dim)
        directions = take(num_samples, directions)

    seen = VectorMemory()
    seen(np.zeros(subdim))

    output_file = opts['--output']
    if output_file:
        if path.exists(output_file):
            old_findings = np.loadtxt(output_file)
            for ray in old_findings:
                seen(ray)
        out = open(output_file, 'a', buffering=1)
    else:
        out = sys.stdout

    rays = (find_xray(lp, v) for v in directions)
    rays = (ray for ray in rays if not seen(ray))

    for ray in rays:
        print(format_vector(ray), file=out)


if __name__ == '__main__':
    hull = main()
