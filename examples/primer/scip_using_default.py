import argparse
import json
from milptune.scip.solver import solve_milp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
        description='Runs incumbent parameter configuration on MILP instances')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument("instance_file", type=str,
        help="MILP instance file")
    
    args = parser.parse_args()

    params = None

    print(args.instance_file)
    _, cost, _ = solve_milp(params, args.instance_file)
    print(cost, args.instance_file)
    with open('runs_default.csv', 'a') as f:
        f.write(str(cost))
        f.write(', ')
        f.write(args.instance_file)
        f.write('\n')
