
from functools import partial
import os
import time

from docopt import docopt
import yaml

from .geom import ConvexCone
from .io import System, SystemFile, StatusInfo, default_column_labels
from .symmetry import SymmetryGroup
from .util import cachedproperty


def main_func(func, doc, args=None):
    app = Application(args, func)
    if doc:
        app.doc = doc
    app.start_timer()
    result = app.run()
    app.stop_timer()
    info_file = app.opts.get('--info')
    if info_file:
        info = {
            'start': app.start,
            'stop': app.stop,
            'time': app.stop - app.start,
            'subspace': app.opts['--subspace'],
        }
        if app.summary:
            info.update(app.summary())
        with open(info_file, 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)
    return result


def application(func, doc=None):
    main = partial(main_func, func, doc)
    if func.__globals__['__name__'] == '__main__':
        main()
    if func.__doc__ is None:
        func.__doc__ = func.__globals__['__doc__'] if doc is None else doc
    return main


class Application:

    def __init__(self, args=None, func=None):
        self.args = args
        self.func = func
        self.start = None
        self.stop = None
        self.summary = None

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
        return ConvexCone.from_cone(self.system, self.subdim, self.limit)

    @cachedproperty
    def output(self):
        return SystemFile(
            self.opts['--output'],
            append=self.resume,
            columns=self.system.columns[:self.subdim],
            pretty=self.pretty,
            symmetries=self.symm_spec)

    @cachedproperty
    def symm_spec(self):
        if self.opts.get('--symmetry') is not None:
            from .symmetry import parse_symmetries
            return parse_symmetries(self.opts['--symmetry'])
        return self.system.symmetries

    @property
    def pretty(self):
        return self.opts.get('--pretty')

    @property
    def symmetries(self):
        col_names = self.system.columns[:self.subdim]
        return SymmetryGroup.load(self.symm_spec, col_names)

    @cachedproperty
    def recursions(self):
        return int(self.opts['--recursions']) or -1

    def report_nullspace(self):
        for face in self.polyhedron.nullspace_int():
            self.report_facet(face)
            self.report_facet(-face)

    @cachedproperty
    def verbosity(self):
        return self.opts['--verbose'] - self.opts['--quiet']

    def info(self, level=0):
        if level + self.verbosity < 0:
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

    def start_timer(self):
        self.start = time.time()

    def stop_timer(self):
        if self.stop is None:
            self.stop = time.time()
