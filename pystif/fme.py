"""
Perform a simple Fourier-Motzkin-Elimination (FME) on the rows.

Usage:
    fme INPUT [-o OUTPUT] -s SUBSPACE [-q] [-d DROP]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -q, --quiet                     No status output
    -d DROP, --drop DROP            Randomly drop DROP inequalities from the
                                    initial system before beginning the FME
                                    [default: 0]
"""

import random
import numpy as np
from docopt import docopt
from .core.fme import FME
from .core.io import System, SystemFile, StatusInfo, default_column_labels


class FMEStatusInfo:

    def __init__(self):
        self.info = StatusInfo()

    def cb_start(self, rows, cols_to_eliminate):
        num_cols = len(rows[0])
        self.info("Eliminate: {} -> {} columns ({} rows)\n"
                  .format(num_cols, num_cols - len(cols_to_eliminate),
                          len(rows)))

    def cb_step(self, rows, col, z, p, n):
        z, p, n = len(z), len(p), len(n)
        num_cols = len(rows[0])
        self.info("  {:3}, z={:4}, p+n={:3}, p*n={:4}"
                  .format(num_cols, z, p+n, p*n))

    def cb_stop(self, rows):
        self.info("")


class VerboseFME(FMEStatusInfo, FME):
    pass


def main(args=None):
    opts = docopt(__doc__, args)

    system = System.load(opts['INPUT'])
    dim = system.dim
    if not system.columns:
        system.columns = default_column_labels(dim)
    system, subdim = system.prepare_for_projection(opts['--subspace'])

    if opts['--quiet']:
        fme = FME()
    else:
        fme = VerboseFME()

    random.seed()
    num_drop = int(opts['--drop'])
    for i in range(num_drop):
        d = random.randrange(system.shape[0])
        system.matrix = np.delete(system.matrix, d, axis=0)

    rows = fme.solve_to(system.matrix, subdim)

    output = SystemFile(opts['--output'], columns=system.columns[:subdim])
    for row in rows:
        output(row)


if __name__ == '__main__':
    main()
