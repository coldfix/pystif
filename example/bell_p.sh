#! /bin/bash

echo Tripartite bell scenario in probability space 
echo Two binary measurements per party

# The results show that there are no non-trivial inequalities on the
# target subspace.

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

subs="AB Ab aB ab AC Ac aC ac BC Bc bC bc"
symm="Aa <> aA; AaBb <> BbAa; BbCc <> CcBb"

# create system of elemental inequalities:
makesys -b "A a B b C c" -o init.txt

# methods other than AFI are too slow...
time afi init.txt -s "$subs" -o fin-sym.txt -q -y "$symm"

# consistency check
equiv init.txt $data/init-bell3x2s.txt
equiv fin-sym $data/final-bell3x2s-2margs.txt

pretty fin-sym -y "$sym"
