#! /bin/bash

echo Cyclic 1D CCA with 3 cells

# For testing inside Travis:
set -e
set -x

here=$(dirname $BASH_SOURCE)
code=323-1
data=$here/$code

# minimize the initial system:
time minimize $data-hum.txt -o min.txt
equiv min.txt $data-hum.txt
equiv min.txt $data-ini.txt

# use different elimination methods:
time chm min.txt -s 7 -o fin-chm.txt
time fme min.txt -s 7 -o fin-fme.txt
time afi min.txt -s 7 -o fin-afi.txt -q
time afi min.txt -s 7 -o fin-sym.txt -q -y "ABCabc <> BCAbca; ABCabc <> CBAcba"

# consistency check
equiv fin-chm.txt $data-fin.txt
equiv fin-fme.txt $data-fin.txt
equiv fin-afi.txt $data-fin.txt
equiv fin-sym.txt $data-fin.txt

pretty fin-chm.txt -y "ABC <> BCA"
