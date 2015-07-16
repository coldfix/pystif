#! /bin/bash

echo Cyclic 1D CCA with 4 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# create system of elemental inequalities and specify some
# structural consraints manually:
makesys -e -v "a b c d A B C D" -o raw.txt \
    "A+B+C+D - ABCD = 0" \
    "aAB + bcdABCD - AB - abcdABCD = 0" \
    "bBC + acdABCD - BC - abcdABCD = 0" \
    "cCD + abdABCD - CD - abcdABCD = 0" \
    "dAD + abcABCD - AD - abcdABCD = 0" \

# DISABLED FOR TEST PERFORMANCE:
# minimize the initial system:
# time minimize raw.txt -o min.txt
cp $data/init-4-1.txt min.txt

# use different elimination methods:
time chm min.txt -s 16 -o fin-chm.txt
time fme min.txt -s 16 -o fin-fme.txt
time afi min.txt -s 16 -o fin-afi.txt \
    -y "abcdABCD <> bcdaBCDA; abcdABCD <> dcbaDCBA"

# consistency check
equiv min.txt $data/init-4-1.txt

equiv fin-chm.txt $data/final-4-1.txt -e
equiv fin-fme.txt $data/final-4-1.txt -e
equiv fin-afi.txt $data/final-4-1.txt -e

pretty fin-chm.txt -y "abcd <> bcda; abcd <> dcba"
