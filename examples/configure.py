import argparse
import json

from milptune.configurator.knn import get_configuration_parameters
from milptune.version import VERSION


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter,
                                     description='Suggests configuration parameters for SCIP')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version',
                        version=f'MILPTune v{VERSION}', help='Shows program\'s version number and exit')
    parser.add_argument('-o', '--output_file', type=str, default='milptune_config.json',
                        help='Specifies the output file to write configuration parameters to')
    parser.add_argument('instance', type=open,
                        help='Path to the instance.mps.gz file')
    parser.add_argument('dataset_name', type=open,
                        help='Name of the dataset the instance belongs to')
    args = parser.parse_args()

    config = get_configuration_parameters(args.instance, args.dataset_name)

    with open(args.output_file, 'w') as f:
        json.dump(config, f)
