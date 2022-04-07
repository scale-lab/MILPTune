list()
{
    for i in $1*.mps.gz
    do
        echo $i
    done
}

list $1 | xargs -i --max-procs=20 python run_smac.py {} $2