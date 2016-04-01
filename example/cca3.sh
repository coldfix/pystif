#! /bin/bash

echo Cyclic 1D CCA with 3 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
data=$here/data

# minimize the initial system:
time minimize $here/cca3.txt -o min.txt

# use different elimination methods:
time chm min.txt -s 7 -o fin-chm.txt
time fme min.txt -s 7 -o fin-fme.txt
time afi min.txt -s 7 -o fin-afi.txt -q
time afi min.txt -s 7 -o fin-sym.txt -q -y "ABCabc <> BCAbca; ABCabc <> CBAcba"

# consistency check
equiv min.txt $here/cca3.txt
equiv min.txt $data/init-3-1.txt

equiv fin-chm.txt $data/final-3-1.txt
equiv fin-fme.txt $data/final-3-1.txt
equiv fin-afi.txt $data/final-3-1.txt
equiv fin-sym.txt $data/final-3-1.txt

pretty fin-chm.txt -y "ABC <> BCA"
