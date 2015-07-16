"""
Utilities to generate the symmetries of equations.
"""

import numpy as np

from .util import VectorMemory


class VarPermutation:

    def __init__(self, varmap):
        self.varmap = varmap

    @classmethod
    def from_subst_rule(cls, orig, perm):
        return cls(dict(zip(orig, perm)))

    def permute_vector(self, vector, col_names):
        col_names = ["".join(sorted(col)) for col in col_names]
        def permute_colname(col):
            return "".join(sorted(self.varmap.get(c, c) for c in col))
        # this actually returns the inverse permutation… shouldn't be harmful
        try:
            return vector[[col_names.index(permute_colname(col))
                           for col in col_names]]
        except ValueError:
            return vector


def evaluate_generators(generators, col_names):
    queue = [tuple(range(len(col_names)))]
    seen = set()
    seen.add(queue[0])
    while queue:
        cur_el = queue.pop()
        yield cur_el
        for gen in generators:
            next_el = tuple(gen.permute_vector(np.array(cur_el), col_names))
            if next_el not in seen:
                seen.add(next_el)
                queue.append(next_el)


class SymmetryGroup:

    def __init__(self, permutations):
        self.permutations = permutations

    def __call__(self, vector):
        seen = VectorMemory()
        for permutation in self.permutations:
            permuted = vector[list(permutation)]
            if not seen(permuted):
                yield permuted

    @classmethod
    def load(cls, spec, col_names):
        generators = [
            VarPermutation.from_subst_rule(*gspec.split('<>'))
            for gspec in spec.replace(" ", "").split(';')
        ]
        return cls(list(evaluate_generators(generators, col_names)))


def NoSymmetry(vector):
    yield vector
