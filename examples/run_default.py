import argparse

import pyscipopt

from pymongo import MongoClient
from pyscipopt import Model as SCIPModel


from milptune.version import VERSION


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


def solve_milp(instance):
    model = SCIPModel()
    model.readProblem(instance)
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam('limits/time', 15 * 60)
    model.hideOutput()

    model.optimize()

    # solution
    sol = model.getBestSol()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    time = model.getSolvingTime()

    return primal - dual, time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter, \
        description='Solves the given instance using default SCIP parameters')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version', \
        version = f'MILPTune v{VERSION}', help='Shows program\'s version number and exit')
    parser.add_argument('instance', type=str, \
        help='Path to the problem instance')
    parser.add_argument('dataset_name', type=str, \
        help='Dataset name in the DB to add these runs to')
    args = parser.parse_args()

    print('connecing to db ..')
    uri = "mongodb://%s:%s@%s:%s" % ('milptune', 'MILP*tune2023', '20.25.127.142', 31331)
    client = MongoClient(uri)
    db = client.milptunedb
    dataset = db[args.dataset_name]
    print('connected to db ..')

    cost, time = solve_milp(args.instance)

    r = dataset.find_one({'path': args.instance})
    defaut_config = {
        'cost': cost,
        'time': time
    }
    update_result = dataset.update_one(r, {'$push': {'default_config': defaut_config}})
    print(args.instance, update_result.modified_count)
