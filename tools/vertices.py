"""
Usage:
    vertices INPUT [-o OUTPUT]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
"""

from pystif.chm import ConvexHull
from pystif.core.app import application
from pystif.core.array import scale_to_int


@application
def main(app):
    ineqs = app.system.matrix
    hull = ConvexHull(ineqs)
    for v in hull.equations:
        app.output(scale_to_int(v))
