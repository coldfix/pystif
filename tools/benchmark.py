"""
Compare runtimes of AFI/CHM by projecting the given input systems to random
subspaces of various dimensions.

Usage:
    benchmark INPUT... [-o OUTPUT] [-s DIMS] [-n NUM] [-l LOGS]

Options:
    -o OUTPUT, --output OUTPUT      Output file
    -s DIMS, --subspaces DIMS       Subspace dimensions     [default: 5..12]
    -n NUM, --num-runs NUM          Runs for each dimension [default: 10]
    -l LOGS, --logs LOGS            Log file folder         [default: ./tmp]
"""

from collections import namedtuple
from functools import partial
import random
import time
import os
import subprocess
import sys

from docopt import docopt
import yaml

from pystif.core.io import System


def main(argv=None):
    opts = docopt(__doc__, argv)
    input_files = opts['INPUT']
    output_dims = sorted(set(parse_range(opts['--subspaces'])))
    num_runs = int(opts['--num-runs'])
    logdir = opts['--logs']
    os.makedirs(logdir, exist_ok=True)
    tasks = make_tasks(input_files, output_dims, num_runs, logdir+'/')
    output_file = _open_for_writing(opts['--output'])
    print_ = partial(_print, file=output_file)
    for task in tasks:
        exec_task(task, print_)


class Task(namedtuple('Task', ['filename', 'system', 'subspace', 'i', 'prefix'])):

    @property
    def name(self):
        return '{}-{}-{}'.format(self.basename, len(self.subspace), self.i)

    @property
    def basename(self):
        return os.path.splitext(os.path.basename(self.filename))[0]


def get_random_subspace(dim, subdim):
    return random.sample(range(dim), subdim)


def make_tasks(input_files, output_dims, num_runs, prefix):
    input_systems = [System.load(f) for f in input_files]
    return (
        Task(filename, system, get_random_subspace(system.dim, subdim), i, prefix)
        for filename, system in zip(input_files, input_systems)
        for subdim in output_dims
        for i in range(num_runs))


def exec_task(task, print_):
    results = (
        *single_pass(task, 'afi', '-r1'),
        *single_pass(task, 'chm'),
        #*single_pass(task, 'fme'),
    )

    # Extract geometrical info on the resulting polytope. This needs to be
    # done separately from the measured AFI call, since evaluating the facial
    # structure can take considerable time:
    summary_file = task.prefix + task.name + '-summary.yml'
    single_pass(task, 'afi', '-r1', '-i', summary_file)
    with open(summary_file) as f:
        summary = yaml.safe_load(f)

    num_rows, dim = task.system.shape

    print_(
        dim, num_rows, len(task.subspace), *results,
        summary['num_facets'],
        summary['num_ridges'],
        summary['num_vertices'],
        summary['ridges_per_facet'][0],
        summary['vertices_per_facet'][0],
        summary['vertices_per_ridge'][0],
    )


def single_pass(task, method, *cmd_args):
    space = " ".join(map(str, task.subspace))
    outf = '{}{}-{}.txt'.format(task.prefix, task.name, method)
    logf = '{}{}-{}.log'.format(task.prefix, task.name, method)
    with open(logf, 'w') as log:
        argv = [
            method, '-q',
            '-s', space,
            task.filename,
            '-o', outf,
            *cmd_args
        ]
        print("\nCommand: {}".format(argv), file=log)
        start = time.time()
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        end = time.time()
        delta = end - start
        print("\nFinished in: {} seconds".format(delta), file=log)
    return delta, proc.returncode


# utility functions

def parse_range(spec):
    parts = spec.split(',')
    for p in parts:
        try:
            start, end = p.split('..')
        except ValueError:
            yield int(p)
            continue
        yield from range(int(start), int(end)+1)


def _open_for_writing(filename):
    if filename and filename != '-':
        return open(filename, 'w')
    return sys.stdout


def _fmt(arg):
    return '{:.3f}'.format(arg) if isinstance(arg, float) else arg


def _print(*args, **kwargs):
    return print(*map(_fmt, args), **kwargs)


if __name__ == '__main__':
    main()
