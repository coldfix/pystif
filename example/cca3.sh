#! /bin/bash

echo Cyclic 1D CCA with 3 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# create system of elemental inequalities and specify some
# structural consraints manually:
makesys -e -v 6 -o raw.txt \
    "D+E+F - DEF <= 0" \
    "ADE + BCDEF - DE - ABCDEF = 0" \
    "BEF + ACDEF - EF - ABCDEF = 0" \
    "CDF + ABDEF - DF - ABCDEF = 0"

# minimize the initial system:
time minimize raw.txt -o min.txt

# use different elimination methods:
time chm min.txt -s 8 -o fin-chm.txt
time fme min.txt -s 8 -o fin-fme.txt
time afi min.txt -s 8 -o fin-afi.txt
time afi min.txt -s 8 -o fin-sym.txt -y "ABCDEF <> BCAEFD"

# consistency check
equiv min.txt raw.txt
equiv min.txt $data/init-3-1.txt

equiv fin-chm.txt $data/final-3-1.txt
equiv fin-fme.txt $data/final-3-1.txt
equiv fin-afi.txt $data/final-3-1.txt
equiv fin-sym.txt $data/final-3-1.txt

pretty fin-chm.txt -y "ABC <> BCA"
