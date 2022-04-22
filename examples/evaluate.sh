list()
{
    for i in $1*.mps.gz
    do
        echo $i
    done
}
N_CORES=$(nproc --all)
FREE_CORES=8
PROCS=$(($N_CORES - $FREE_CORES))
list $1 | xargs -i --max-procs=$PROCS python evaluate.py {} $2 eval
