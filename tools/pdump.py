#! /usr/bin/env python
"""
Print profiling results.

Usage:
    pdump STATS [-c COL] [-n NUM]

Options:
    -c COL, --column COL        sort by column [default: cumulative]
    -n NUM, --num-lines NUM     number of lines [default: 15]

Valid column names:
    calls 	        call count
    cumulative 	    cumulative time
    cumtime 	    cumulative time
    file 	        file name
    filename 	    file name
    module 	        file name
    ncalls 	        call count
    pcalls 	        primitive call count
    line 	        line number
    name 	        function name
    nfl 	        name/file/line
    stdname 	    standard name
    time 	        internal time
    tottime 	    internal time
"""

import pstats
from docopt import docopt


def main(args=None):
    opts = docopt(__doc__, args)
    stats_file = opts['STATS']
    sort_col = opts['--column']
    num_lines = int(opts['--num-lines'])
    stats = pstats.Stats(stats_file)
    stats.strip_dirs().sort_stats(sort_col).print_stats(num_lines)


if __name__ == '__main__':
    main()
