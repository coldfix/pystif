
import sys
from warnings import warn


def raw_cca(nf, nl, ni):
    if nl ==  0: warn("Trivial CCA: no causal constraints")
    if nl ==  1: warn("Trivial CCA: not enough parents -> IID final state")
    if nl >= ni: warn("Trivial CCA: too many parents -> no causal constraints")

    overlap = (nf+nl-1)-ni
    if overlap <   0: warn("Unused initial cells.")
    if overlap >= ni: warn("Too few initial variables!")

    yield "# CCA with {} i.i.d. cells, each influencing {} of {} dependend cells".format(ni, nf, nl)
    yield "# Cyclic overlap: {}".format(overlap)

    alphabet = "".join(chr(ord("a") + i) for i in range(26))
    ALPHABET = alphabet.upper()

    ini = alphabet[:ni]
    fin = ALPHABET[:nf]

    if nl == 0 or nf <= 1:
        return

    yield "rvar " + " ".join(fin + ini)
    yield ""
    yield "# iid initial state"
    yield " :: ".join(ini)
    yield ""
    yield "# causality DAG"
    for i, v in enumerate(fin):
        ii = ini[i:] + ini[:i]
        Nd = ii[nl:]
        Pa = ii[:nl]
        Nd = fin.replace(v, "") + Nd
        yield "{} :: {} | {}".format(v, ",".join(Nd), ",".join(Pa))


def main(args):
    nums = list(map(int, args[0]))
    print("\n".join(raw_cca(*nums)))


if __name__ == '__main__':
    main(sys.argv[1:])
