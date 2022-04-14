list()
{
    for i in $1*.mps.gz
    do
        echo $i
    done
}

list $1 | xargs -i --max-procs=24 python run_default.py {} $2