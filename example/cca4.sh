#! /bin/bash

echo Cyclic 1D CCA with 4 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# DISABLED FOR TEST PERFORMANCE:
# minimize the initial system:
# time minimize cca4.txt -o min.txt
cp $data/init-4-1.txt min.txt

# use different elimination methods:
time chm min.txt -s 15 -o fin-chm.txt -q
time fme min.txt -s 15 -o fin-fme.txt -q
time afi min.txt -s 15 -o fin-afi.txt -q -r1 \
    -y "abcdABCD <> bcdaBCDA; abcdABCD <> dcbaDCBA"

# consistency check
equiv min.txt $data/init-4-1.txt

equiv fin-chm.txt $data/final-4-1.txt -e
equiv fin-fme.txt $data/final-4-1.txt -e
equiv fin-afi.txt $data/final-4-1.txt -e

pretty fin-chm.txt -y "abcd <> bcda; abcd <> dcba"
