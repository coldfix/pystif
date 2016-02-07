"""
Print equations in human readable format.

Usage:
    pretty INPUT [-o OUTPUT] [-y SYM] [-q]

Options:
    -o OUTPUT, --output OUTPUT      Output file
    -y SYM, --symmetry SYM          Specify symmetry group generators
    -q, --quiet                     Less output in the presence of symmetries
"""

from docopt import docopt

from .core.io import System, print_to, default_column_labels
from .core.symmetry import SymmetryGroup, group_by_symmetry


def _fmt_float(f):
    if round(f) == f:
        return str(int(f))
    return str(f)


def _coef(coef):
    if coef < 0:
        prefix = '-'
        coef = -coef
    else:
        prefix = '+'
    if coef != 1:
        prefix += ' ' + _fmt_float(coef)
    return prefix


def format_human_readable(constraint, columns):
    lhs = [(_coef(c), columns[i], c > 0)
           for i, c in enumerate(constraint)
           if c != 0]
    # len() is used as approximation for number of terms involved. For most
    # cases this should be fine.
    lhs = sorted(lhs, key=lambda term: (term[2], len(term[1])),
                 reverse=True)
    lhs = ["{} {}".format(coef, col) for coef, col, _ in lhs]
    if not lhs:
        lhs = ["0"]
    return "{} ≥ 0".format(" ".join(lhs).lstrip('+ '))


def main(args=None):
    opts = docopt(__doc__, args)
    system = System.load(opts['INPUT'])
    if system.columns:
        columns = system.columns
    else:
        columns = default_column_labels(system.dim)
    print_ = print_to(opts['--output'])
    def dump(rows):
        for row in rows:
            print_(format_human_readable(row, columns))

    symmetries = opts['--symmetry']
    if symmetries is None:
        symmetries = system.symmetries
    if symmetries:
        sg = SymmetryGroup.load(symmetries, system.columns)
        groups = group_by_symmetry(sg, system.matrix)
        for g in groups:
            if opts['--quiet']:
                dump([g[0]])
            else:
                dump(g)
                print_()
        lengths = list(map(len, groups))
        print_()
        print_('# Number of inequalities:')
        print_('#   groups:', len(groups))
        print_('#   items: ', *lengths)
        print_('#   total: ', sum(lengths))
    else:
        dump(system.matrix)


if __name__ == '__main__':
    main()
