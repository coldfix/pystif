#! /bin/bash

echo Cyclic 1D CCA with 4 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
code=424-1
data=$here/$code

# DISABLED FOR TEST PERFORMANCE:
# # minimize the initial system:
# time minimize $data-hum.txt -o min.txt
# equiv min.txt $data-ini.txt
cp $data-ini.txt min.txt

# use different elimination methods:
time chm min.txt -s 15 -o fin-chm.txt -q
time fme min.txt -s 15 -o fin-fme.txt -q
time afi min.txt -s 15 -o fin-afi.txt -q \
    -y "abcdABCD <> bcdaBCDA; abcdABCD <> dcbaDCBA"

equiv fin-chm.txt $data-fin.txt
equiv fin-fme.txt $data-fin.txt
equiv fin-afi.txt $data-fin.txt

pretty fin-chm.txt -y "abcd <> bcda; abcd <> dcba"
