#! /bin/bash

echo Tripartite bell scenario with 2 measurements per party

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

subs="_AB _Ab _aB _ab _AC _Ac _aC _ac _BC _Bc _bC _bc"
symm="Aa <> aA; AaBb <> BbAa; BbCc <> CcBb"

# create system of elemental inequalities:
makesys "rvar A a B b C c L" "A a :: B b :: C c | L" -o init.txt
# H(A₀A₁ B₀B₁ C₀C₁|λ) = H(A₀A₁|λ) + H(B₀B₁|λ) + H(C₀C₁|λ)

# methods other than AFI are too slow...
time afi init.txt -s "$subs" -o fin-sym.txt -q -y "$symm" -r1

# consistency check
equiv init.txt $data/init-bell3x2s.txt
equiv fin-sym $data/final-bell3x2s-2margs.txt

pretty fin-sym -y "$sym"
