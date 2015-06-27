"""
Minimize a system of inequalities.

Usage:
    minimize INPUT [-o OUTPUT] [-e] [-q]

Options:
    -o OUTPUT, --output OUTPUT      Write final (minimized) system to this file
    -a, --append                    Open output file in append mode
    -e, --el-ineqs                  Extend system with elemental inequalities
    -q, --quiet                     No status output (otherwise goes to STDERR)
"""

from docopt import docopt
from .core.lp import Minimize
from .core.io import System, SystemFile, StatusInfo, default_column_labels


class MinimizeStatusInfo:

    def __init__(self):
        self.info = StatusInfo()

    def cb_start(self, rows):
        self.num_orig = len(rows)

    def cb_step(self, idx, rows):
        self.info("Minimizing: {} -> {}  (i={})"
                  .format(self.num_orig, len(rows), idx))


    def cb_stop(self, rows):
        self.info("Minimizing: {} -> {}  (DONE)"
                  .format(self.num_orig, len(rows)))


class VerboseMinimize(MinimizeStatusInfo, Minimize):
    pass


def main(args=None):
    opts = docopt(__doc__, args)

    system = System.load(opts['INPUT'])
    dim = system.dim
    if not system.columns:
        system.columns = default_column_labels(dim)

    if opts['--quiet']:
        m = Minimize()
    else:
        m = VerboseMinimize()

    rows = m.minimize(system.matrix)

    output = SystemFile(opts['--output'], columns=system.columns)
    for row in rows:
        output(row)


if __name__ == '__main__':
    main()
