#! /bin/bash

echo Bipartite bell scenario with 2 measurements per party

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)

# create system of elemental inequalities:
makesys "rvar A a B b" -o init.txt

subs="_AB _Ab _aB _ab _A _a _B _b"
symm="Aa <> aA; AaBb <> BbAa"

# use different elimination methods:
time chm init.txt -s "$subs" -o fin-chm.txt
time fme init.txt -s "$subs" -o fin-fme.txt
time afi init.txt -s "$subs" -o fin-afi.txt -q
time afi init.txt -s "$subs" -o fin-sym.txt -q -y "$symm"

time rfd init.txt -s "$subs" -o fin-rfd.txt -qq -y "$symm"
time rfd init.txt -s "$subs" -o fin-rfd.txt -qq -y "$symm" -d 7

# consistency check
equiv init.txt $here/2x2-ini.txt

equiv fin-chm.txt $here/2x2-fin-08D.txt
equiv fin-fme.txt $here/2x2-fin-08D.txt
equiv fin-afi.txt $here/2x2-fin-08D.txt

pretty fin-chm.txt -y "Aa <> aA; AaBb <> BbAa"
