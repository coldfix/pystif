"""
Project convex cone to subspace by an adjacent facet iteration method.

Usage:
    afi INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-r NUM] [-q]... [-v]... [-p] [-i FILE]

Options:
    -o OUTPUT, --output OUTPUT      Set output file for solution
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -y SYM, --symmetry SYM          Symmetry group generators
    -r NUM, --recursions NUM        Number of AFI recursions [default: 0]
    -q, --quiet                     Show less output
    -v, --verbose                   Show more output
    -p, --pretty                    Pretty print output inequalities
    -i FILE, --info FILE            Print short summary to file (YAML)
"""

from functools import partial

import os
import numpy as np

from .core.app import application
from .core.util import VectorMemory, PointSet
from .core.linalg import matrix_rank
from .chm import convex_hull_method, print_status, print_qhull


def mean_variance(samples):
    return (float(np.mean(samples)),
            float(np.var(samples)),
            float(np.std(samples, ddof=1)))


class Face:

    """
    Stores the currently known local geometric structure of a face. This state
    is computed and altered during the AFI procedure.

    :ivar ConvexCone polyhedron:        Provider for LP primitives
    :ivar LinearSubspace subspace:      Defining r-dimensional subspace
    :ivar list supfaces:                Known (r+1)-dimensional superfaces
    :ivar list subfaces:                Known (r-1)-dimensional subfaces
    :ivar dict adjacent:                facet, subface -> adjacent facet
    :ivar dict normals:                 subface -> normal vector
    :ivar bool solved:                  completely solved
    :ivar bool visited:                 Visited during this AFI run
    :ivar int rank:                     Dimensionality
    """

    def __init__(self, polyhedron):
        self.polyhedron = polyhedron
        self.subspace = polyhedron.subspace()
        self.supfaces = []
        self.subfaces = []
        self.adjacent = {}
        self.normals = {}
        self.solved = False
        self.visited = False
        self.rank = self.subspace.onb.shape[0]

    def add_subface(self, subface, normal):
        """
        Add a subface (initially without known adjacency).

        :param Face subface:    the subface (may be incomplete at this time)
        :param normal:          normal vector of the subface for this face
        """
        subface.supfaces.append(self)
        self.subfaces.append(subface)
        self.normals[subface] = normal

    def unvisit(self):
        """
        Clear visited flag if not fully solved.
        """
        if self.visited and not self.solved:
            self.visited = False
            for f in self.subfaces:
                f.unvisit()

    def is_same_face(self, sup, normal):
        """
        Check if the subface defined by a superface and normal is the same
        as this one.
        """
        normal = np.atleast_2d(normal)
        return (np.allclose(self.subspace.onb @ normal.T, 0) and
                np.allclose(self.subspace.onb @ sup.subspace.normals.T, 0))


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

    def __init__(self, polyhedron, symmetries, recursions, info, verbosity):
        self.polyhedron = polyhedron
        self.symmetries = symmetries
        self.recursions = recursions
        self._verbosity = verbosity
        self._info_file = info
        self.full_rank = polyhedron.rank()
        self._info = info()
        self.whole = Face(polyhedron)
        self.faces = [[] for i in range(self.full_rank)]
        self.faces[-1].append(self.whole)
        # for info summary:
        self._vertices = PointSet()
        self._num_chm = 0

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
        if body.visited:
            yield from self._sol(body)
        elif body.rank == 1:
            yield from self._ray(body)
        elif body.rank == self.full_rank - self.recursions:
            yield from self._chm(body)
        else:
            yield from self._afi(body)
        body.visited = True

    def _sol(self, body):
        """Iterate over all previously discovered subfaces of a polytope."""
        yield from body.normals.items()

    def _ray(self, body):
        """Iterate over the vertices of a 1D face."""
        # NOTE: currently only handling rays originating at zero, so we
        # just need a single additional subface (~vertex):
        normal = body.subspace.onb[0]
        facet = self.get_subface_instance(body, normal)
        body.add_subface(facet, normal)
        body.solved = True
        return [(facet, normal)]

    def _chm(self, body):
        """Iterate over all subfaces of an arbitrary face using CHM."""
        self._num_chm += 1
        sub_info = self._info_file(body.rank - self.full_rank)
        callbacks = (self._vertices.add,
                     lambda facet: None,
                     partial(print_status, sub_info),
                     partial(print_qhull, sub_info))
        poly = body.polyhedron
        ineqs, _ = convex_hull_method(poly, poly.basis(), *callbacks,
                                      symmetries=self.symmetries)
        for ineq in ineqs:
            yield self.get_subface_instance(body, ineq), ineq
        body.solved = True

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
        body.solved = all(face.solved for face in body.subfaces)

    def init_face(self, face):
        """Ensure knowledge of a subface."""
        if face.subfaces:
            subface = face.subfaces[0]
            return subface, face.normals[subface]
        return self._search_subface(face)

    def _search_subface(self, face):
        """
        Search a subface using an LP.

        :returns: a tuple ``(Face, normal)``.
        """
        if face.solved:
            return self.init_face(face)
        rank = face.rank
        self.info(rank, "Search facet of {}D face", rank)
        # TODO: obtain chain of subfaces of dimensions 1 to N in single sweep
        guess = np.ones(self.polyhedron.dim)
        normal = face.polyhedron.refine_to_facet(guess)
        facet = self.get_subface_instance(face, normal)
        self.info(rank, "Search facet of {}D face [done]\n", rank)
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
        rank = face.rank - 1
        for f in self.faces[rank]:
            if f.is_same_face(face, normal):
                face.add_subface(f, normal)
                return f
        polyhed = self._intersect(face, normal)
        subface = Face(polyhed)
        face.add_subface(subface, normal)
        self.faces[rank].append(subface)
        return subface

    def info(self, rank, text, *args, **kwargs):
        if rank - self.full_rank + self._verbosity >= 0:
            self._info(text.format(*args, **kwargs))

    def status_info(self, queue, equations, i, rank):
        num_eqs = len(equations)
        len_eqs = len(str(num_eqs))
        self.info(rank, "AFI rank {} queue: {:4}, progress: {}/{}{}", rank,
                  len(queue), str(i).rjust(len_eqs), num_eqs,
                  "\n" if i == num_eqs else "")

    def summary(self):
        # TODO: make this work in the presence of symmetries!
        facets = [f for f in self.whole.subfaces if f.visited]
        ridges = set()
        facetv = VectorMemory()
        for f in facets:
            symmetries = list(self.symmetries(self.whole.normals[f]))
            f.multiplicity = len(symmetries)
            facetv.add(*symmetries)
        facetv = np.array(list(facetv.seen))
        dim = self.full_rank-1
        def is_vertex(v):
            active = [f for f in facetv if np.isclose(f @ v, 0)]
            return (len(active) >= dim and
                    matrix_rank(active) >= dim)
        vertices = VectorMemory()
        for v in self._vertices:
            v = np.array(v)
            if is_vertex(v):
                vertices.add(*list(self.symmetries(v)))
        vertices = np.array(list(vertices.seen))
        for f in facets:
            for r in f.subfaces:
                if r in ridges:
                    continue
                # if any(is_face_identical(r, _r) for _r in ridges):
                #     continue
                ridges.add(r)
        # TODO: other statistics like variance of quantities:
        # - vertex/ridge
        # - vertex/face
        # - ridge/face
        rpf = mean_variance([
            len(f.subfaces) for f in facets
        ])
        vpf = mean_variance([
            sum(1 for v in vertices if np.allclose(f.subspace.normals @ v, 0))
            for f in facets
        ])
        vpr = mean_variance([
            sum(1 for v in vertices if np.allclose(r.subspace.normals @ v, 0))
            for r in ridges
        ])
        return {
            'num_facets': len(facets),
            'num_ridges': len(ridges),
            'num_vertices': len(vertices),
            'num_chm': self._num_chm,
            'ridges_per_facet': rpf,
            'vertices_per_facet': vpf,
            'vertices_per_ridge': vpr,
        }


def is_face_identical(a, b):
    return a is b or np.allclose(a.subspace.onb @ b.subspace.normals.T, 0)


@application
def main(app):
    app.report_nullspace()
    app.start_timer()
    app.verbosity += 1
    afi = AFI(app.polyhedron, app.symmetries, app.recursions, app.info,
              app.verbosity)
    for facet in afi.solve():
        app.output(facet)
    afi.info(afi.full_rank, "")
    app.summary = afi.summary
