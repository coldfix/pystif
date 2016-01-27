"""
Utilities to generate the symmetries of equations.
"""

import numpy as np

from .util import VectorMemory, scale_to_int


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


class Permutation:

    def __init__(self, p):
        self.p = list(p)

    def __call__(self, vector):
        return vector[self.p]

    def __len__(self):
        return len(self.p)

    def inverse(self):
        result = [None] * len(self.p)
        for i, v in enumerate(self.p):
            result[v] = i
        return self.__class__(result)


class SymmetryGroup:

    def __init__(self, permutations):
        self.permutations = list(permutations)

    def __iter__(self):
        return iter(self.permutations)

    def __call__(self, vector):
        seen = VectorMemory()
        for permutation in self.permutations:
            permuted = permutation(vector)
            if not seen(permuted):
                yield permuted

    @classmethod
    def load(cls, spec, col_names):
        generators = [
            VarPermutation.from_subst_rule(*gspec.split('<>'))
            for gspec in spec.replace(" ", "").split(';')
        ]
        return cls(map(Permutation, evaluate_generators(generators, col_names)))


def NoSymmetry(vector):
    yield vector


def group_by_symmetry(sg, vectors):
    groups = []
    belong = {}
    for row in vectors:
        row_ = tuple(scale_to_int(row))
        if row_ in belong:
            belong[row_].append(row)
            continue
        group = [row]
        groups.append(group)
        for sym in sg(row):
            sym_ = tuple(scale_to_int(sym))
            belong[sym_] = group
    return groups
