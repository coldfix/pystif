#! /bin/bash

echo Bipartite bell scenario with 2 measurements per party

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# create system of elemental inequalities:
makesys -e -v "A a B b" -o init.txt

subs="AB Ab aB ab A a B b"
symm="Aa <> aA; AaBb <> BbAa"

# use different elimination methods:
time chm init.txt -s "$subs" -o fin-chm.txt
time fme init.txt -s "$subs" -o fin-fme.txt
time afi init.txt -s "$subs" -o fin-afi.txt -q
time afi init.txt -s "$subs" -o fin-sym.txt -q  -y "$symm"

time rfd init.txt -s "$subs" -o fin-rfd.txt -y "$symm"
time rfd init.txt -s "$subs" -o fin-rfd.txt -y "$symm" -d 7

# consistency check
equiv init.txt $data/init-bell2x2.txt

equiv fin-chm.txt $data/final-bell2x2.txt
equiv fin-fme.txt $data/final-bell2x2.txt
equiv fin-afi.txt $data/final-bell2x2.txt

pretty fin-chm.txt -y "Aa <> aA; AaBb <> BbAa"
