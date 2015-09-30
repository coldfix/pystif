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

from .core.app import application
from .core.lp import Minimize
from .core.io import StatusInfo


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


@application
def main(app):
    m = Minimize() if app.quiet else VerboseMinimize()
    for row in m.minimize(app.system.matrix):
        app.output(row)
