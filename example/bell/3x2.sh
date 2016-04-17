#! /bin/bash

echo Tripartite bell scenario with 2 measurements per party

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)

subs="_AB _Ab _aB _ab _AC _Ac _aC _ac _BC _Bc _bC _bc"
symm="Aa <> aA; AaBb <> BbAa; BbCc <> CcBb"

# create system of elemental inequalities:
makesys "rvar A a B b C c L" "A a :: B b :: C c | L" -o init.txt
# H(A₀A₁ B₀B₁ C₀C₁|λ) = H(A₀A₁|λ) + H(B₀B₁|λ) + H(C₀C₁|λ)

# methods other than AFI are too slow...
time afi init.txt -s "$subs" -o fin-sym.txt -q -y "$symm"


# time fme init.txt -s "_ABC _ABc _AbC _Abc _aBC _aBc _abC _abc _AB _Ab _aB _ab _AC _Ac _aC _ac _BC _Bc _bC _bc _A _a _B _b _C _c"

# consistency check
equiv init.txt $here/3s2-ini.txt
equiv fin-sym.txt $here/3x2-fin-12D.txt

pretty fin-sym.txt -y "$sym"
