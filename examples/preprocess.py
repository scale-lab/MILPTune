import argparse
import pathlib
import pickle

from bson.binary import Binary
from pymongo import MongoClient

from milptune.features.A import get_A, get_mapping
from milptune.version import VERSION


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter, \
        description='Preprocesses training data for MILPTune')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version', \
        version = f'MILPTune v{VERSION}', help='Shows program\'s version number and exit')
    parser.add_argument('-o', '--output_dir', type=str, default='processed', \
        help='Specifies the output directory to write processed data to')
    parser.add_argument('data_dir', type=str, \
        help='Path to the data directory that contains train and valid folders')
    args = parser.parse_args()

    instances_path = pathlib.Path(args.data_dir)
    training_instances = list(instances_path.glob('train/*.mps.gz'))
    validation_instances = list(instances_path.glob('valid/*.mps.gz'))

    dataset_name = instances_path.stem
    client = MongoClient(host='20.232.144.167')
    db = client.milptunedb
    dataset = db[dataset_name]

    # primer to just focus on common matrix coeficients
    vars_index, conss_index = get_mapping(training_instances[0])

    for instance in training_instances:
        A = get_A(instance, vars_index, conss_index)
        dataset.insert_one(
            {
                'path': str(instance),
                'A': Binary(pickle.dumps(A, protocol=4))
            }
        )
