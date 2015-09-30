
from docopt import docopt
from functools import wraps
import os

from .geom import ConvexPolyhedron
from .io import System, SystemFile, StatusInfo
from .linalg import addz
from .symmetry import SymmetryGroup, NoSymmetry
from .util import cachedproperty


def application(func, doc=None):
    def main(args=None):
        app = Application(args, func)
        if doc:
            app.doc = doc
        return app.run()
    if func.__globals__['__name__'] == '__main__':
        main()
    return main


class Application:

    def __init__(self, args=None, func=None):
        self.args = args
        self.func = func

    @cachedproperty
    def doc(self):
        return self.func.__doc__ or self.func.__globals__['__doc__']

    @cachedproperty
    def opts(self):
        return docopt(self.doc, self.args)

    @cachedproperty
    def system(self):
        system = System.load(self.opts['INPUT'])
        if not system.columns:
            system.columns = default_column_labels(self.dim)
        try:
            subspace = self.opts['--subspace']
        except KeyError:
            subdim = system.dim
        else:
            system, subdim = system.prepare_for_projection(subspace)
        system.subdim = subdim
        return system

    @property
    def subdim(self):
        return self.system.subdim

    @cachedproperty
    def dim(self):
        return self.system.dim

    @cachedproperty
    def resume(self):
        return self.opts.get('--resume')

    @cachedproperty
    def limit(self):
        return float(self.opts['--limit'])

    @cachedproperty
    def polyhedron(self):
        return ConvexPolyhedron.from_cone(self.system, self.subdim, self.limit)

    @cachedproperty
    def output(self):
        return SystemFile(
            self.opts['--output'],
            append=self.resume,
            columns=self.system.columns[:self.subdim])

    @property
    def symmetries(self):
        if self.opts.get('--symmetry'):
            col_names = self.system.columns[:self.subdim]
            return SymmetryGroup.load(self.opts['--symmetry'], col_names)
        else:
            return NoSymmetry

    @cachedproperty
    def recursions(self):
        return int(self.opts['--recursions']) or -1

    def report_nullspace(self):
        for face in addz(self.polyhedron.subspace().normals):
            self.report_facet(face)
            self.report_facet(-face)

    @cachedproperty
    def quiet(self):
        return self.opts['--quiet']

    def info(self, level=0):
        if self.quiet > level:
            return StatusInfo(open(os.devnull, 'w'))
        else:
            return StatusInfo()

    def report_facet(self, facet):
        for sym in self.symmetries(facet):
            self.output(sym)

    def run(self):
        if self.func:
            return self.func(self)
        raise NotImplementedError()
