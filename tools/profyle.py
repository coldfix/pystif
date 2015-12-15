#! /usr/bin/env python
"""
Small profiling utility for pystif modules. (Note that the script name is
chosen to avoid a name clash with the builtin 'profile' module.)

Usage:
    profyle.py [-o FILE] -- [ARGV...]

Arguments:
    ARGV        Command to execute. The first argument must be the
                name of a pystif module, e.g. 'afi' or 'chm'

Options:
    -h, --help                      Help
    -o FILE, --output FILE          Output file
"""

import cProfile
from importlib import import_module
import itertools
import os

from docopt import docopt


def run_module():
    module = import_module(mod_name)
    module.main(cmd_args)


def new_file(basename, ext):
    for i in itertools.count():
        filename = "{}.{}{}".format(basename, i, ext)
        if not os.path.exists(filename):
            return filename


def main(args=None):
    opts = docopt(__doc__, args)

    # Need to set these as globals, because of weird cProfile.run API which
    # doesn't accept code objects with "free variables".
    global mod_name
    global cmd_args

    mod_name = "pystif." + opts['ARGV'][0]
    cmd_args = opts['ARGV'][1:]
    stats_file = opts['--output'] or new_file(mod_name, ".stats")

    cProfile.run(run_module.__code__, stats_file)


if __name__ == '__main__':
    main()
