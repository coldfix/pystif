"""
Project convex cone to subspace by an adjacent facet iteration method.

Usage:
    afi INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-r NUM] [-q]...

Options:
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -y SYM, --symmetry SYM          Symmetry group generators
    -r NUM, --recursions NUM        Number of AFI recursions [default: 0]
    -q, --quiet                     Show less output
"""

from functools import partial

import os
import numpy as np
from docopt import docopt

from .core.app import application
from .core.array import scale_to_int
from .core.io import StatusInfo
from .core.util import VectorMemory
from .chm import convex_hull_method, print_status, print_qhull


class Face:

    """
    Stores the currently known local geometric structure of a face. This state
    is computed and altered during the AFI procedure.

    :ivar ConvexCone polyhedron:        Provider for LP primitives
    :ivar LinearSubspace subspace:      Defining subspace
    :ivar list subfaces:                Known subfaces
    :ivar dict adjacent:                facet, subface -> adjacent facet
    :ivar dict normals:                 subface -> normal vector (relative to first supface)
    :ivar bool solved:                  completely solved
    """

    def __init__(self, polyhedron):
        self.polyhedron = polyhedron
        self.subspace = polyhedron.subspace()
        self.subfaces = []
        self.adjacent = {}
        self.normals = {}
        self.solved = False
        self.rank = self.subspace.onb.shape[0]

    def add_subface(self, subface, normal):
        """
        Add a subface (initially without known adjacency).

        :param Face subface:    the subface (may be incomplete at this time)
        :param normal:          normal vector of the subface for this face
        """
        self.subfaces.append(subface)
        self.normals[subface] = normal


class AFIQueue:

    def __init__(self, afi):
        self.afi = afi
        self.queue = []
        self.seen = VectorMemory()

    def __len__(self):
        return len(self.queue)

    def pop(self):
        return self.queue.pop()

    def add(self, facet, normal):
        if self.seen(normal):
            return
        yield (facet, normal)
        self.queue.append(facet)
        for symm in self.afi.symmetries(normal):
            self.seen(symm)


class AFI:

    """
    Performs AFI (Adjacent Facet Iteration) and stores the result.
    """

    def __init__(self, polyhedron, symmetries, recursions, quiet_rank, info):
        self.polyhedron = polyhedron
        self.symmetries = symmetries
        self.recursions = recursions
        self.quiet_rank = quiet_rank
        self.full_rank = polyhedron.rank()
        self._info = info
        self.whole = Face(polyhedron)

    def solve(self):
        """Iterate over facet normal vectors of the overall polyhedron."""
        for facet, normal in self._solve_face(self.whole):
            yield from self.symmetries(normal)

    def _solve_face(self, body):
        """
        Iterate over essentially different subfaces (:class:`Face`) of `body`.

        :param Face body: abstract face graph
        :param ConvexCone polyhedron: face realization subspace
        """
        if body.solved:
            yield from self._sol(body)
        elif body.rank == 1:
            yield from self._ray(body)
        elif body.rank == self.full_rank - self.recursions:
            yield from self._chm(body)
        else:
            yield from self._afi(body)
        body.solved = True

    def _sol(self, body):
        """."""
        yield from body.normals.items()

    def _ray(self, body):
        """Iterate over the vertices of a 1D face."""
        # NOTE: currently only handling rays originating at zero, so we
        # just need a single additional subface (~vertex):
        normal = body.subspace.onb[0]
        facet = self.get_subface_instance(body, normal)
        body.add_subface(facet, normal)
        return [(facet, normal)]

    def _chm(self, body):
        """Iterate over all subfaces of an arbitrary face using CHM."""
        if body.rank <= self.quiet_rank:
            sub_info = StatusInfo(open(os.devnull, 'w'))
        else:
            sub_info = self._info
        callbacks = (lambda ray: None,
                     lambda facet: None,
                     partial(print_status, sub_info),
                     partial(print_qhull, sub_info))
        poly = body.polyhedron
        ineqs, _ = convex_hull_method(poly, poly.basis(), *callbacks)
        for ineq in ineqs:
            yield self.get_subface_instance(body, ineq), ineq

    def _afi(self, body):
        """Iterate over all subfaces of an arbitrary face using AFI."""
        queue = AFIQueue(self)
        yield from queue.add(*self.init_face(body))
        while queue:
            facet = queue.pop()
            subfaces = list(self._solve_face(facet))
            for i, (subface, _) in enumerate(subfaces):
                self.status_info(queue, subfaces, i, body.rank)
                yield from queue.add(*self.get_adjacent_face(body, facet, subface))
            self.status_info(queue, subfaces, len(subfaces), body.rank)

    def init_face(self, face):
        """Ensure knowledge of a subface."""
        if face.subfaces:
            subface = face.subfaces[0]
            return subface, face.normals[subface]
        rank = face.rank-1
        self.info(rank, "Search rank {} facet", rank)
        # TODO: obtain chain of subfaces of dimensions 1 to N in single sweep
        guess = np.ones(self.polyhedron.dim)
        normal = face.polyhedron.refine_to_facet(guess)
        facet = self.get_subface_instance(face, normal)
        self.info(rank, "Search rank {} facet [done]\n", rank)
        return facet, normal

    def get_adjacent_face(self, sup, face, sub):
        try:
            f = sup.adjacent[face, sub]
            return f, sup.normals[f]
        except KeyError:
            pass

        g0 = sup.normals[face]
        s0 = face.normals[sub]
        g, s = sup.polyhedron.get_adjacent_facet(g0, s0)
        adjf = self.get_subface_instance(sup, g)

        adjf.add_subface(sub, s)

        sup.adjacent[adjf, sub] = face
        sup.adjacent[face, sub] = adjf
        return adjf, g

    def _intersect(self, face, normal):
        polyhedron = face.polyhedron.intersection(normal)
        polyhedron._subspace = face.subspace.add_normals(normal)
        polyhedron._rank = face.rank - 1
        return polyhedron

    def get_subface_instance(self, face, normal):
        """
        Get/instanciate subface object with the given normal vector.

        The returned object is guaranteed to be a registered subface of the
        given face.

        :param Face face: parental face
        :param np.ndarray normal: subface normal vector in canonical basis
        """
        for f, n in face.normals.items():
            if np.allclose(n, normal):
                return f
        polyhed = self._intersect(face, normal)
        subface = Face(polyhed)
        face.add_subface(subface, normal)
        return subface

    def info(self, rank, text, *args, **kwargs):
        if rank >= self.quiet_rank:
            self._info(text.format(*args, **kwargs))

    def status_info(self, queue, equations, i, rank):
        num_eqs = len(equations)
        len_eqs = len(str(num_eqs))
        self.info(rank, "AFI rank {} queue: {:4}, progress: {}/{}{}", rank,
                  len(queue), str(i).rjust(len_eqs), num_eqs,
                  "\n" if i == num_eqs else "")


@application
def main(app):
    app.report_nullspace()
    quiet_rank = app.subdim-2 if app.quiet else 0
    info = app.info(1)
    afi = AFI(app.polyhedron, app.symmetries, app.recursions, quiet_rank, info)
    for facet in afi.solve():
        app.output(facet)
    info()
