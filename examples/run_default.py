import argparse

from pymongo import MongoClient

from milptune.version import VERSION
from milptune.scip.solver import solve_milp
from milptune.db.connections import get_client


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


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
    client = get_client()
    db = client.milptunedb
    dataset = db[args.dataset_name]
    print('connected to db ..')

    cost, time = solve_milp(instance=args.instance)

    r = dataset.find_one({'path': args.instance})
    defaut_config = {
        'cost': cost,
        'time': time
    }
    update_result = dataset.update_one(r, {'$push': {'default_config': defaut_config}})
    print(args.instance, update_result.modified_count)
