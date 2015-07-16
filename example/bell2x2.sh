#! /bin/bash

echo Bipartite bell scenario with 2 measurements per party

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# create system of elemental inequalities:
makesys -e -v "A a B b" -o init.txt

# use different elimination methods:
chm init.txt -s "AB Ab aB ab A a B b" -o fin-chm.txt
fme init.txt -s "AB Ab aB ab A a B b" -o fin-fme.txt
afi init.txt -s "AB Ab aB ab A a B b" -o fin-afi.txt
afi init.txt -s "AB Ab aB ab A a B b" -o fin-sym.txt \
    -y "Aa <> aA; AaBb <> BbAa"

# consistency check
equiv init.txt $data/init-bell2x2.txt

equiv fin-chm.txt $data/final-bell2x2.txt
equiv fin-fme.txt $data/final-bell2x2.txt
equiv fin-afi.txt $data/final-bell2x2.txt

pretty fin-chm.txt -y "Aa <> aA; AaBb <> BbAa"
