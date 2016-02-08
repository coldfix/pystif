"""
Print equations in human readable format.

Usage:
    pretty INPUT [-o OUTPUT] [-y SYM] [-q]

Options:
    -o OUTPUT, --output OUTPUT      Output file
    -y SYM, --symmetry SYM          Specify symmetry group generators
    -q, --quiet                     Less output in the presence of symmetries
"""

from .core.app import application


@application
def main(app):

    system = app.system
    output = app.output

    if system.symmetries:
        short = app.opts['--quiet']
        groups = output.pprint_symmetries(system.matrix, short)
        lengths = list(map(len, groups))
        print_ = output._print
        print_()
        print_('# Number of inequalities:')
        print_('#   groups:', len(groups))
        print_('#   items: ', *lengths)
        print_('#   total: ', sum(lengths))
    else:
        output.pprint(system.matrix)


if __name__ == '__main__':
    main()
