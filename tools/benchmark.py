"""
Compare runtimes of AFI/CHM by projecting the given input systems to random
subspaces of various dimensions.

Usage:
    benchmark INPUT [-o OUTPUT] [-s DIMS] [-n NUM] [-l LOGS]

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

COLUMNS = (
    (' d',          'out_dim'),
    ('   t_afi',    'afi_time'),
    ('   t_chm',    'chm_time'),
    ('   f',        'num_facets'),
    ('   r',        'num_ridges'),
    ('   v',        'num_vertices'),
    ('     r_f',    'ridges_per_facet'),
    ('     v_f',    'vertices_per_facet'),
    ('     v_r',    'vertices_per_ridge'),
    ('   _tafi',    'afi_symm_time'),
    ('   _tchm',    'chm_symm_time'),
    ('  _f',        'fafi_symm_num_facets'),
    ('  _r',        'afi_symm_num_ridges'),
    ('  _v',        'afi_symm_num_vertices'),
)


def main(argv=None):
    opts = docopt(__doc__, argv)
    input_file = opts['INPUT']
    output_dims = sorted(set(parse_range(opts['--subspaces'])))
    num_runs = int(opts['--num-runs'])
    logdir = opts['--logs']
    os.makedirs(logdir, exist_ok=True)
    system = System.load(input_file)
    tasks = make_tasks(system, input_file, output_dims, num_runs, logdir+'/')
    output_file = _open_for_writing(opts['--output'])
    print_ = partial(_print, file=output_file, flush=True)
    print_('# input system:')
    print_('#  - file:', input_file)
    print_('#  - rows:', system.shape[0])
    print_('#  - dim: ', system.shape[1])
    cols_short, cols_long = zip(*COLUMNS)
    print_('#', *cols_long)
    print_()
    print_('#', *cols_short)
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


def make_tasks(system, filename, output_dims, num_runs, prefix):
    return (
        Task(filename, system, get_random_subspace(system.dim, subdim), i, prefix)
        for subdim in output_dims
        for i in range(num_runs))


def exec_task(task, print_):
    afi_pure = single_pass(task, 'afi', 'pure', '-y', '', '-r1')
    chm_pure = single_pass(task, 'chm', 'pure', '-y', '')
    afi_symm = single_pass(task, 'afi', 'symm', '-r1')
    chm_symm = single_pass(task, 'chm', 'symm')

    if not afi_pure or not chm_pure or not afi_symm or not chm_symm:
        print_(
            len(task.subspace),
            bool(afi_pure),
            bool(chm_pure),
            bool(afi_symm),
            bool(chm_symm),
        )
        return
    print_(
        len(task.subspace),
        afi_pure['time'],
        chm_pure['time'],
        afi_pure['num_facets'],
        afi_pure['num_ridges'],
        afi_pure['num_vertices'],
        afi_pure['ridges_per_facet'][0],
        afi_pure['vertices_per_facet'][0],
        afi_pure['vertices_per_ridge'][0],
        afi_symm['time'],
        chm_symm['time'],
        afi_symm['num_facets'],
        afi_symm['num_ridges'],
        afi_symm['num_vertices'],
    )


def single_pass(task, method, extra, *cmd_args):
    space = " ".join(map(str, task.subspace))
    basename = '{}{}-{}-{}'.format(task.prefix, task.name, extra, method)
    outf = basename + '.txt'
    logf = basename + '.log'
    inff = basename + '.yml'
    errf = basename + '.err'
    with open(logf, 'w') as log, open(errf, 'w') as err:
        argv = [
            method, '-q',
            '-s', space,
            task.filename,
            '-o', outf,
            '-i', inff,
            *cmd_args
        ]
        print("\nCommand: {}".format(argv), file=log)
        start = time.time()
        proc = subprocess.run(argv, stdout=log, stderr=err)
        end = time.time()
        delta = end - start
        print("\nFinished in: {} seconds".format(delta), file=log)
    if proc.returncode != 0:
        return None
    with open(inff) as f:
        return yaml.safe_load(f)


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
    if isinstance(arg, float):
        return '{:8.3f}'.format(arg)
    if isinstance(arg, int):
        return '{:4}'.format(arg)
    return arg


def _print(*args, **kwargs):
    return print(*map(_fmt, args), **kwargs)


if __name__ == '__main__':
    main()
