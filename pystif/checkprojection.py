"""
Check a set of constraints if they are facets of the projection cone.

Usage:
    checkprojection SYSTEM CHECK [-y SYMM]

Arguments:
    SYSTEM                      The original system
    CHECK                       File with candidate faces of the projection cone

Options:
    -y SYM, --symmetry SYM      Symmetry group generators
"""

from docopt import docopt

from .core.geom import ConvexCone
from .core.io import format_vector, System
from .core.symmetry import group_by_symmetry


def check_face(cone, ineq):
    is_face = cone.is_face(ineq)
    rank = cone.face_rank(ineq) if is_face else 0
    return (is_face, rank)


def check_faces(cone, check):

    results = [(ineq, *check_face(cone, ineq)) for ineq in check]

    facetrank = cone.rank()-1

    invalid = [ineq for ineq, is_face, rank in results
               if not is_face]
    subface = [(ineq, facetrank-rank) for ineq, is_face, rank in results
               if is_face and rank < facetrank]

    if invalid:
        print("The following are not valid faces:")
        for ineq in invalid:
            print(ineq)
    else:
        print("All given inequalities are valid faces!")

    if subface:
        for ineq, missrank in subface:
            print(ineq, "is a", -missrank, "face")
    else:
        print("All valid faces are facets!")

    return not invalid and not subface


def main(args=None):
    opts = docopt(__doc__)

    system = System.load(opts['SYSTEM'])
    check = System.load(opts['CHECK'])

    system, subdim = system.slice(check.columns, fill=True)

    if not check.symmetries:
        check.update_symmetries(system.symmetries)
    check.update_symmetries(opts['--symmetry'])

    groups = group_by_symmetry(check.symmetry_group(), check.matrix)
    ineqs = [g[0] for g in groups]
    cone = ConvexCone.from_cone(system, subdim, 1)

    return 0 if check_faces(cone, ineqs) else 1


if __name__ == '__main__':
    sys.exit(main())
