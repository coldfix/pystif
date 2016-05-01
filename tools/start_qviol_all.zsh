#! /usr/bin/env zsh

# ts = task spooler
ts -S 4

data=example/bell
outp=example/qviol-bell3x2

function start_qviol() {
    conedim=$1
    subdims=$2
    constrs=(${@:3})

    facets=$data/3x2-fin-$conedim.txt
    prefix=$outp/$conedim-$subdims
    noconstr=$prefix-none.yml

    echo "Starting $@"
    job_num=$(ts qviol $facets -d $subdims -o $noconstr)
    echo $job_num

    for constr in $constrs; do
        ts -D $job_num qviol $noconstr \
            -d $subdims \
            -o $prefix-$constr.yml \
            -c $constr \
            -p ${parametrization-all}
    done
}

function start_all() {
    if [[ -n $@ ]]; then
        dims=($@)
    else
        dims=(26D 18D 14D 12D 08D)
    fi

    for dim in $dims; do
        start_qviol $dim 333 CHSHE CGLMP PPT
        start_qviol $dim 222 CHSHE CHSH PPT
    done
}
