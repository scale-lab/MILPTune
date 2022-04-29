import argparse
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from milptune.version import VERSION


def distances(actual, estimated, output_file):
    plt.clf()
    # sns.set_theme()

    # g = sns.scatterplot(x=estimated, y=actual, legend="full")
    scatter = plt.scatter(estimated, actual)


    plt.plot(range(0, 100), range(0, 100), color='silver', linestyle='dashed')
    # plt.plot([100] * 120, range(0, 120), color='silver', linestyle='dashed')
    # plt.plot(range(0, 120), [100] * 120, color='silver', linestyle='dashed')

    plt.xlabel("Estimated Cost", fontsize=15)
    plt.ylabel("Actual Cost", fontsize=15)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.grid()
    # plt.xticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 'No Sol.', ''])
    # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 'No Sol.', ''])
    # plt.legend(bbox_to_anchor=(0, 1.2), loc='upper left', ncol=2)
    plt.savefig(output_file, bbox_inches='tight')
    # plt.show()


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter, \
        description='Suggests configuration parameters for SCIP')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version', \
        version = f'MILPTune v{VERSION}', help='Shows program\'s version number and exit')
    parser.add_argument('validation_dir', type=str, \
        help='Specifies the validation dir that has .csv files')
    args = parser.parse_args()
    
    validation_path = pathlib.Path(args.validation_dir)
    instances = list(validation_path.glob('*.csv'))

    actual = []
    estimated = []
    for instance in instances:       
        with open(instance, 'r') as f:
            a, e = [], []
            for line in f:
                source, rank, estimated_cost, actual_cost, time, _, distance = line.strip().split(';')
                if source == 'milptune':
                    a.append(float(actual_cost))
                    e.append(float(estimated_cost))
            actual.append(min(a))
            estimated.append(min(e))
                
    distances(actual, estimated, f'distance.pdf')


