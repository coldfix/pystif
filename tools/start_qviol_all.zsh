#! /usr/bin/env zsh

# ts = task spooler
ts -S 4

data=example/data

function start_qviol() {
    conedim=$1
    subdims=$2
    constrs=(${@:3})

    facets=$data/final-bell3x2-$conedim.txt
    prefix=$data/qviol-bell3x2-$conedim-$subdims
    noconstr=$prefix-none.yml

    echo "Starting $@"
    job_num=$(ts qviol $facets -d $subdims -o $noconstr)
    echo $job_num

    for constr in $constrs; do
        ts -D $job_num qviol $noconstr -d $subdims -o $prefix-$constr.yml -c $constr
    done
}

if [[ -n $@ ]]; then
    dims=($@)
else
    dims=(26D 18D 14D 12D 08D)
fi

for dim in $dims; do
    start_qviol $dim 333 CHSHE CGLMP
    start_qviol $dim 222 CHSHE CHSH SEP
done

# ts qviol $data/final-bell3x2-08D.txt -d 222 -o $data/qviol-bell3x2-08D-222-none.yml
# ts qviol $data/final-bell3x2-08D.txt -d 333 -o $data/qviol-bell3x2-08D-333-none.yml
# ts qviol $data/final-bell3x2-12D.txt -d 222 -o $data/qviol-bell3x2-12D-222-none.yml
# ts qviol $data/final-bell3x2-12D.txt -d 333 -o $data/qviol-bell3x2-12D-333-none.yml
# ts qviol $data/final-bell3x2-18D-rfd.txt -d 222 -o $data/qviol-bell3x2-18D-222-none.yml
# ts qviol $data/final-bell3x2-18D-rfd.txt -d 333 -o $data/qviol-bell3x2-18D-333-none.yml
# ts qviol $data/final-bell3x2-26D-p2f.txt -d 222 -o $data/qviol-bell3x2-26D-222-none.yml
# ts qviol $data/final-bell3x2-26D-p2f.txt -d 333 -o $data/qviol-bell3x2-26D-333-none.yml


# ----------------------------------------
# prefix=example/data/qviol-bell3x2
#
# # 08D
# ts qviol ${prefix}-08D-222-none.yml -d 222 -o ${prefix}-08D-222-chshe.yml -c CHSHE
# ts qviol ${prefix}-08D-222-none.yml -d 222 -o ${prefix}-08D-222-chsh.yml -c CHSH
# ts qviol ${prefix}-08D-222-none.yml -d 222 -o ${prefix}-08D-222-ppt.yml -c SEP
#
# ts qviol ${prefix}-08D-333-none.yml -d 333 -o ${prefix}-08D-333-chshe.yml -c CHSHE
# ts qviol ${prefix}-08D-333-none.yml -d 333 -o ${prefix}-08D-333-cglmp.yml -c CGLMP
#
# # 12D
# ts qviol ${prefix}-12D-222-none.yml -d 222 -o ${prefix}-12D-222-chshe.yml -c CHSHE
# ts qviol ${prefix}-12D-222-none.yml -d 222 -o ${prefix}-12D-222-chsh.yml -c CHSH
# ts qviol ${prefix}-12D-222-none.yml -d 222 -o ${prefix}-12D-222-ppt.yml -c SEP
#
# ts qviol ${prefix}-12D-333-none.yml -d 333 -o ${prefix}-12D-333-chshe.yml -c CHSHE
# ts qviol ${prefix}-12D-333-none.yml -d 333 -o ${prefix}-12D-333-cglmp.yml -c CGLMP
#
# # 18D
# ts qviol ${prefix}-18D-222-none.yml -d 222 -o ${prefix}-18D-222-chshe.yml -c CHSHE
# ts qviol ${prefix}-18D-222-none.yml -d 222 -o ${prefix}-18D-222-chsh.yml -c CHSH
# ts qviol ${prefix}-18D-222-none.yml -d 222 -o ${prefix}-18D-222-ppt.yml -c SEP
#
# ts qviol ${prefix}-18D-333-none.yml -d 333 -o ${prefix}-18D-333-chshe.yml -c CHSHE
# ts qviol ${prefix}-18D-333-none.yml -d 333 -o ${prefix}-18D-333-cglmp.yml -c CGLMP
#
# # 26D
# ts qviol ${prefix}-26D-222-none.yml -d 222 -o ${prefix}-26D-222-chshe.yml -c CHSHE
# ts qviol ${prefix}-26D-222-none.yml -d 222 -o ${prefix}-26D-222-chsh.yml -c CHSH
# ts qviol ${prefix}-26D-222-none.yml -d 222 -o ${prefix}-26D-222-ppt.yml -c SEP
#
# # ts qviol ${prefix}-26D-333-none.yml -d 333 -o ${prefix}-26D-333-chshe.yml -c CHSHE
# # ts qviol ${prefix}-26D-333-none.yml -d 333 -o ${prefix}-26D-333-cglmp.yml -c CGLMP
# ----------------------------------------
