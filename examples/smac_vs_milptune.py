import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from milptune.version import VERSION


def smac_vs_milptune(smac, milptune, output_file):
    plt.clf()
    sns.set_theme()
    smac = list(map(lambda p: min(p, 100), smac))
    milptune = list(map(lambda p: min(p, 100), milptune))

    for i in range(len(smac)):
        if milptune[i] >= 100:
            milptune[i] = milptune[i] + np.random.uniform(1, 10)
        if smac[i] >= 100:
            smac[i] = smac[i] + np.random.uniform(1, 10)

    hue = []
    label = []
    for i in range(len(smac)):
        if milptune[i] > 100 and smac[i] > 100:
            hue.append('red')
            label.append('No Sol. Both Configs')
        elif milptune[i] < smac[i]:
            hue.append('green')
            label.append('MILPTune Config Better')
        else:
            hue.append('blue')
            label.append('SMAC Config Better')

    sns.scatterplot(x=smac, y=milptune, hue=label, palette=['green', 'cornflowerblue'], legend="full")

    plt.plot(range(0, 100), range(0, 100), color='silver', linestyle='dashed')
    plt.plot([100] * 120, range(0, 120), color='silver', linestyle='dashed')
    plt.plot(range(0, 120), [100] * 120, color='silver', linestyle='dashed')

    plt.xlabel("SMAC Cost", fontsize=15)
    plt.ylabel("MILPTune Cost", fontsize=15)
    plt.xlim([0, 120])
    plt.ylim([0, 120])
    plt.xticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 'No Sol.', ''])
    plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 'No Sol.', ''])
    plt.legend(bbox_to_anchor=(0, 1.2), loc='upper left', ncol=2)
    plt.savefig(output_file, bbox_inches='tight')


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
    parser.add_argument('validation_dir', type=str,
                        help='Specifies the validation dir that has .csv files')
    parser.add_argument('k', type=int, default=1,
                        help='Specifies how many configs to look at (1-5)')
    args = parser.parse_args()

    validation_path = pathlib.Path(args.validation_dir)
    instances = list(validation_path.glob('*.csv'))

    results: dict = {
        'default': [],
        'smac': [],
        'milptune': []
    }
    for instance in instances:
        with open(instance, 'r') as f:
            cost = {
                'default': 1_000_000.0,
                'smac': 1_000_000.0,
                'milptune': 1_000_000.0
            }
            for line in f:
                source, rank, estimated_cost, actual_cost, time, _, _ = line.strip().split(';')
                if float(actual_cost) < cost[source]:
                    if source in ['milptune', 'smac']:
                        if int(rank) in list(range(args.k)):
                            cost[source] = float(actual_cost)
                    else:
                        cost[source] = float(actual_cost)

            for k, v in cost.items():
                results[k].append(v)

    smac_vs_milptune(results['smac'], results['milptune'], f'smac_vs_milptune-{args.k}.pdf')
