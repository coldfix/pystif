"""
Utilities to generate the symmetries of equations.
"""

from functools import reduce
from operator import add

import numpy as np

from .util import VectorMemory, scale_to_int, _name, _varset


def _all_unique(s):
    """Check that all items of s are unique."""
    return len(set(s)) == len(s)


class VarPermutation:

    def __init__(self, varmap):
        self.varmap = varmap
        # Complete incomplete specifications, e.g.
        #   A <> a      ->  (Aa)    = Aa <> aA
        #   ABC <> BDC  ->  (ABD)   = ABD <> BDA
        inverse = {v: k for k, v in varmap.items()}
        for k in list(varmap):
            if k not in inverse:
                # k is the start of an incomplete cycle
                f = varmap[k]
                while f in varmap:
                    f = varmap[f]
                varmap[f] = k

    @classmethod
    def from_subst_rule(cls, orig, perm):
        assert _all_unique(orig)
        assert _all_unique(perm)
        assert len(orig) == len(perm)
        return cls(dict(zip(orig, perm)))

    @classmethod
    def from_cycle_spec(cls, cycles):
        # NOTE: using delayed import as to avoid dependency cycle through:
        #   io -> symmetry -> parse -> io
        from .parse import flatten
        return flatten((c, c[1:]+c[:1]) for c in cycles)

    def permute_colname(self, col):
        return frozenset(self.varmap.get(v, v) for v in _varset(col))

    def permute_vector(self, vector, col_names):
        col_names = [_varset(col) for col in col_names]
        col_index = {n: i for i, n in enumerate(col_names)}
        # this actually returns the inverse permutation… shouldn't be harmful
        try:
            return vector[[col_index[self.permute_colname(col)]
                           for col in col_names]]
        except KeyError:
            raise ValueError(
                "The permutation {} replaces a column that is not included in the list"
                " of column names {}."
                .format(self.varmap, col_names))

    def permute_symbolic(self, vector):
        return {
            _name({self.varmap.get(c, c) for c in _varset(k)}): v
            for k, v in vector.items()
        }


def evaluate_generators(generators, col_names):
    # TODO: check this function…
    queue = [tuple(range(len(col_names)))]
    seen = set()
    seen.add(queue[0])
    while queue:
        cur_el = queue.pop()
        yield cur_el
        for gen in generators:
            try:
                next_el = tuple(gen.permute_vector(np.array(cur_el), col_names))
            except ValueError:
                continue
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
        if not spec:
            return cls([Permutation(range(len(col_names)))])
        if isinstance(spec, str):
            spec = parse_symmetries(spec)
        generators = [VarPermutation.from_subst_rule(a, b)
                      for a, b in spec]
        return cls(map(Permutation, evaluate_generators(generators, col_names)))

    @classmethod
    def from_cycles(cls, cycles, col_names):
        return cls.load(cycles_to_spec(cycles), col_names)


def cycles_to_spec_item(cycles):
    cycles = list(cycles)
    if not cycles:
        return (), ()
    a = reduce(add, cycles)
    b = reduce(add, (c[1:] + c[:1] for c in cycles))
    return a, b


def cycles_to_spec(rules):
    return [cycles_to_spec_item(cycles) for cycles in rules]


def parse_symmetries(s):
    if not s.strip():
        return ()
    from .parse import symm_list, tokenize
    return symm_list.parse(tokenize(s))


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
