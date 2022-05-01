import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from milptune.version import VERSION


def bin_count(default, smac, milptune, output_file):
    plt.clf()
    mpl.rcParams['xtick.major.pad'] = 0

    sns.set_theme(style="white", context="talk")

    # Set up the matplotlib figure
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

    # Generate some sequential data
    bins = np.arange(0, 100, 10)
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f'{bins[i]}-{bins[i+1]}')
    labels.append('No Sol.')

    y3 = np.bincount(np.digitize(milptune, bins), minlength=10)
    sns.barplot(x=labels, y=y3[1:], palette="rocket", ax=ax1)
    ax1.bar_label(ax1.containers[0], fontsize=10)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel("MILPTune", fontsize=15)

    y2 = np.bincount(np.digitize(smac, bins), minlength=10)
    sns.barplot(x=labels, y=y2[1:], palette="rocket", ax=ax2)
    ax2.bar_label(ax2.containers[0], fontsize=10)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("SMAC", fontsize=15)

    y1 = np.bincount(np.digitize(default, bins), minlength=10)
    sns.barplot(x=labels, y=y1[1:], palette="rocket", ax=ax3)
    ax3.bar_label(ax3.containers[0], fontsize=10)
    ax3.axhline(0, color="k", clip_on=False)
    ax3.set_ylabel("Default", fontsize=15)
    ax3.set_xlabel("Optimization Cost", fontsize=15)

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    plt.setp(ax3.get_xticklabels(), fontsize=10)
    plt.xticks(rotation=85)

    plt.savefig(output_file)


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

    bin_count(results['default'], results['smac'], results['milptune'], f'top-{args.k}.pdf')
