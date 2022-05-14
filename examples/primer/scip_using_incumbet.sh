list()
{
    for i in $1*.mps.gz
    do
        echo $i
    done
}

list $1 | xargs -i --max-procs=25 python scip_using_incumbent.py {}