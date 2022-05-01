import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from milptune.version import VERSION


def distances(similarity_distances, method, output_file):
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 4))
    # ax.set_xscale("log")
    df = pd.DataFrame({
        'distance': similarity_distances,
        'method': method
    })
    df.sort_values(by='method', inplace=True, ascending=False)

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x='distance', y='method', data=df,
                whis=[0, 98], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x="distance", y="method", data=df,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="", xlabel='KNN Distance')
    sns.despine(trim=True, left=True)
    plt.tight_layout()
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
    args = parser.parse_args()

    validation_path = pathlib.Path(args.validation_dir)
    instances = list(validation_path.glob('*.csv'))

    similarity_distances = []
    methods = []
    for instance in instances:
        with open(instance, 'r') as f:
            d = []
            min_cost = 1_000_000
            method = None
            for line in f:
                source, rank, estimated_cost, actual_cost, time, _, distance = line.strip().split(';')
                if source == 'milptune':
                    d.append(float(distance))
                    if float(actual_cost) < min_cost:
                        min_cost = float(actual_cost)
                        method = 'MILPTune'
                elif source == 'smac':
                    if float(actual_cost) < min_cost:
                        min_cost = float(actual_cost)
                        method = 'smac'
            if method == 'MILPTune':
                similarity_distances.append(min(d))
                methods.append('MILPTune')
            elif method == 'smac':
                similarity_distances.append(max(d))
                methods.append('SMAC')

    distances(similarity_distances, methods, 'distance_switch.pdf')
