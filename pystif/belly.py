"""
Usage:
    belly NUM_PARTIES NUM_CE
"""

import itertools

from .core.app import application


def partitions(n: int, max_terms: int):
    """
    Iterate all partitions of ``n``, i.e. all positive integer tuples that
    sum up to ``n``.

    The partitions are obtained in descending order.

    Params:
        n: the number to be partitioned
        max_terms: maximum number of terms in a partition

    Example:

        >>> list(partitions(3, 3))
        [(3,), (2, 1), (1, 1, 1)]

        >>> list(partitions(4, 2))
        [(4,), (3, 1), (2, 2)]
    """
    def _partitions(n, max_, max_terms):
        if n == 0:
            yield ()
            return
        if max_terms * max_ < n:
            return
        if max_terms == 1:
            yield (n,)
            return
        for k in range(min(n, max_), 0, -1):
            for p in _partitions(n-k, k, max_terms-1):
                yield (k, *p)
    return _partitions(n, n, max_terms)


class Partitioner:

    def partitions(self, n: int):
        """
        Iterate all partitions of ``n``, i.e. all positive integer tuples that
        sum up to ``n``.

        The partitions are obtained in ascending order.

        Params:
            n: the number to be partitioned
        """
        if n == 0:
            yield ()
            return
        if self.max_n < n:
            return
        if self.max_terms == 1:
            yield (n,)
            return
        for k in range(min(n, self.max_k)+1):
            for p in self.next(k).partitions(n-k):
                yield (k, *p)


class ComboPartitioner(Partitioner):

    """
    Helper tool to iterate all partitions of a number.

    Each term in a partition corresponds to a pairing of two symbols (hi, lo).
    There is a defined number of times that each value of lo or hi must occur
    in the resulting partition.
    """

    def __init__(self, compat, hi, lo):
        self.compat = compat
        self.hi = hi
        self.lo = lo
        self.max_terms = len(self.compat)
        count = [min(self.hi.get(h, 0), -self.lo.get(l, 0))
                 for h, l in self.compat]
        self.max_k = count[0]
        # this is just an upper bound, but that's ok:
        self.max_n = sum(count)

    def next(self, k):
        """
        Set value for first component to ``k`` and return a partitioner for the
        remaining components.
        ."""
        if k == 0:
            return self.__class__(self.compat[1:], self.hi, self.lo)
        (h, l), *compat = self.compat
        hi = self.hi.copy()
        lo = self.lo.copy()
        d_add(hi, h, -k)
        d_add(lo, l, -k)
        return self.__class__(compat, hi, lo)


def d_add(d, k, v):
    """Add a number to a dict item."""
    d.setdefault(k, 0)
    d[k] += v
    if d[k] == 0:
        del d[k]


def l_del(l, i):
    """Return a copy of the list/tuple with the i'th element removed."""
    return l[:i] + l[i+1:]


def merged(a, b):
    """Return the merge of two dictionaries."""
    r = {}
    r.update(a)
    r.update(b)
    return r


# TODO: how to prove thhat a face is a facet?
#   - generate rays + check dimensionality
#   - check minimality of its elemental form decomposition


VarSet = "VarSet = tuple[int] - a sorted set of variable indices."
Expression = """
    Expression = {VarSet: int} - mapping of coefficients for the Shannon
    entropy term corresponding to the given subset of random variables.
"""


def _iter_ineqs_cmi(terms_hi: VarSet, terms_lo: VarSet, len_hi: int) -> Expression:

    """
    This function recursively marginalizes an entropy expression of the form

        Σ c_h H(X_h) - Σ c_l H(X_l)

    by inserting CMIs (conditional mutual informations) until there are only
    terms with at most 2-body entropies left. ``h`` and ``l`` are subsets with
    n and (n-1) elements respectively and it must hold that

        Σ c_h = Σ c_l,

    and that they can be matched one-to-one in a subset-compatible way.

    ``terms_hi``, ``terms_lo`` hold the coefficients of the current
    expansion state.
    """

    # The expansion will stop when arriving at an expression with only Shannon
    # entropy terms with at most ``max_vars`` variables, e.g. ``max_vars=2``
    # means that we get only the two-body marginals H(AB), H(Ab), … in the
    # result expressions.
    max_vars = 2

    # This case be entered only for ``num_parties=1`` which is kind of
    # irrelevant, but I'll keep it for clarity anyway.
    if len_hi == max_vars:
        yield merged(terms_hi, terms_lo)
        return

    # The last expansion step is free to ignore the combinations with
    # ``terms_lo`` - arbitrary subsets can be chosen from the hi terms:
    if len_hi == max_vars+1:

        # paremetrization for chosing the subsets for each individual term:
        subset_choices = list(itertools.product(
            range(len_hi),
            range(len_hi-1)
        ))

        impls = itertools.product(*(
            itertools.combinations_with_replacement(subset_choices, n)
            for h, n in terms_hi.items()
        ))

        for rm_elems in impls:

            # We can simply forget terms_hi, since these will all be
            # compensated in the following. terms_lo must be kept since those
            # terms are ignored for chosing the final expansion step.
            new_hi = terms_lo.copy()
            new_lo = {}

            for h, sels in zip(terms_hi, rm_elems):

                for ia, ib in sels:

                    # Add the CMI I(a:b|c) = H(ac) + H(bc) - H(c) - H(abc)
                    # to compensate for the H(abc) term in terms_hi:

                    abc = h
                    ac = l_del(abc, ib+(ib>=ia))
                    bc = l_del(abc, ia)
                    c = l_del(ac, ia)

                    d_add(new_hi, ac, 1)
                    d_add(new_hi, bc, 1)
                    d_add(new_lo, c, -1)

            yield merged(new_hi, new_lo)

        return

    # TODO: check if this structural constraint is valid
    compat = [
        (hi, lo)
        for hi, lo in itertools.product(terms_hi, terms_lo)
        if set(lo) <= set(hi)
    ]

    ctrl = ComboPartitioner(compat, terms_hi, terms_lo)
    for part in ctrl.partitions(sum(terms_hi.values())):

        impls = itertools.product(*(
            # The pair (h, l) together with one additional parameter selecting
            # one element to remove from l is enough to specify the CMI:
            itertools.combinations_with_replacement(range(len(l)), n)
            for (h, l), n in zip(compat, part)
        ))

        for rm_elems in impls:

            # We can ignore both terms_lo and terms_hi, since all those terms
            # will be compensated for by chosing the below CMIs:
            new_hi = {}
            new_lo = {}

            for (h, l), sels in zip(compat, rm_elems):

                for ib in sels:
                    a = set(h) - set(l)
                    b = {l[ib]}
                    c = set(l) - b
                    # I(a:b|c) = H(ac) + H(bc) - H(c) - H(abc)
                    d_add(new_hi, tuple(sorted(a|c)), 1)
                    d_add(new_lo, tuple(sorted(c)), -1)

            yield from _iter_ineqs_cmi(new_hi, new_lo, len_hi-1)


def iter_bell_ineqs(num_parties, num_ce):

    """
    Iterate inequalities with at most two-body terms.

    Params:
        num_parties: number of variables
        num_ce: total number of conditional entropy terms
    """

    num_vars = 2 * num_parties
    varlist = tuple(range(num_vars))

    # Loop over all possible combinations of conditional entropies
    for part in partitions(num_ce, num_vars):
        # Every partition ``p = (p_0, p_1, ...)`` where ``Σ p_i = num_ce``
        # means that the conditional entropy ``H(i|X) = H(iX) - H(X)`` occurs
        # ``p_i`` times:
        terms_hi = {varlist: num_ce}
        terms_lo = {l_del(varlist, i): -c for i, c in enumerate(part)}
        for ineq in _iter_ineqs_cmi(terms_hi, terms_lo, num_vars):
            # TODO: keep track of all previously seen ineqs to avoid multiple
            # re-expansion.
            yield from assign_parties(ineq, num_parties)


def assign_parties(terms: VarSet, num_parties: int):

    """
    """

    affiliations = itertools.product(*(
        range(2*i-1) for i in range(num_parties, 0, -1)
    ))

    for aff in affiliations:
        avail = list(range(2*num_parties))
        parties = []
        for p in aff:
            first = avail.pop(0)
            second = avail.pop(p)
            parties.append((first, second))

        mod_terms = terms.copy()
        queue = set(mod_terms)
        while queue:
            t = queue.pop()
            n = mod_terms.get(t, 0)
            if n == 0:
                continue
            for a, b in parties:
                if a in t and b in t:
                    # This expands terms of the form H(A,a,X) via I(A:a|X).
                    # TODO: also use expansions of the form I(A:a|X).

                    abc = t
                    ac = l_del(abc, abc.index(b))
                    bc = l_del(abc, abc.index(a))
                    c = l_del(ac, ac.index(a))

                    # I(a:b|c) = H(ac) + H(bc) - H(c) - H(abc)
                    d_add(mod_terms, abc, -n)
                    d_add(mod_terms, ac, +n)
                    d_add(mod_terms, bc, +n)
                    d_add(mod_terms, c, -n)

                    queue.add(ac)
                    queue.add(bc)
                    queue.add(c)

                    # TODO: if we allowed for more than 3-body terms, there
                    # can be multiple possible orders to remove parties here,
                    # i.e.: H(AaBbXx) ->
                    #   I(A:a|BbXx) then eliminate Bb (or vice versa) or
                    #   I(A:X|aBbx) then eliminate Bb (or vice versa) or
                    #   I(A:B|abXx)
                    break

        yield mod_terms


@application
def main(app):
    opts = app.opts

    num_parties = int(opts['NUM_PARTIES'])
    num_ces = [int(x) for x in opts['NUM_CE'].split(',')]

    for num_ce in num_ces:
        for ineq in iter_bell_ineqs(num_parties, num_ce):
            print(ineq)
