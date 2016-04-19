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
    (' vpf_geo',    'vertices_per_facet_geom'),
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
    passes = (
        single_pass(task, 'afi', '-y', '', '-r1'),
        single_pass(task, 'chm', '-y', ''),
    )
    afi, chm = passes

    dim = len(task.subspace)
    if not all(passes):
        print_(
            dim,
            bool(afi),
            bool(chm),
        )
        return
    vpf_detail = afi['vertices_per_facet_detail']
    vpf_geom = (sum(n**((dim-1)/2) for n in vpf_detail)/len(vpf_detail)) ** (2/(dim-1))
    print_(
        dim,
        afi['time'],
        chm['time'],
        afi['num_facets'],
        afi['num_ridges'],
        afi['num_vertices'],
        afi['ridges_per_facet'][0],
        afi['vertices_per_facet'][0],
        afi['vertices_per_ridge'][0],
        vpf_geom,
    )


def single_pass(task, method, *cmd_args):
    space = " ".join(map(str, task.subspace))
    basename = '{}{}-{}'.format(task.prefix, task.name, method)
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
