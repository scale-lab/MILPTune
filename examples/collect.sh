list()
{
    for i in {1..10000000}
    do
        echo $i
    done
}
N_CORES=$(nproc --all)
FREE_CORES=2
PROCS=$(($N_CORES - $FREE_CORES))
list | xargs -i --max-procs=$PROCS python run_smac.py $1