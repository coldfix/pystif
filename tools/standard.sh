
mkdir -p bench/std/logs

source tools/sub.zsh

ts -S 4

sys=example/data/init-bell3x2.txt
log=bench/std/logs

# 08D

ts afi $sys -s $sub_08 -i $log/08D-afi-symm.yml -o $log/08D-afi-symm
ts chm $sys -s $sub_08 -i $log/08D-chm-symm.yml -o $log/08D-chm-symm

ts afi $sys -s $sub_08 -i $log/08D-afi-pure.yml -o $log/08D-afi-pure -y ''
ts chm $sys -s $sub_08 -i $log/08D-chm-pure.yml -o $log/08D-chm-pure -y ''

# 12D

ts afi $sys -s $sub_12 -i $log/12D-afi-symm.yml -o $log/12D-afi-symm
ts chm $sys -s $sub_12 -i $log/12D-chm-symm.yml -o $log/12D-chm-symm

ts chm $sys -s $sub_12 -i $log/12D-chm-pure.yml -o $log/12D-chm-pure -y ''
ts afi $sys -s $sub_12 -i $log/12D-afi-pure.yml -o $log/12D-afi-pure -y ''
