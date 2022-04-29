import argparse
import json
import os
from multiprocessing import Process

from milptune.configurator.knn import get_configuration_parameters
from milptune.scip.solver import solve_milp
from milptune.smac.hpo import optimize
from milptune.version import VERSION


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


def incumbent():
    return {
        "branching/clamp": 0.21765904731140243,
        "branching/lpgainnormalize": "d",
        "branching/midpull": 0.12642028100595432,
        "branching/midpullreldomtrig": 0.8501486164786376,
        "branching/preferbinary": True, 
        "branching/scorefac": 0.6247093852526081, 
        "branching/scorefunc": "q", 
        "lp/colagelimit": 390219615, 
        "lp/pricing": "f", 
        "lp/rowagelimit": 441481087, 
        "nodeselection/childsel": "i", 
        "separating/cutagelimit": 374707679, 
        "separating/maxcuts": 483812320, 
        "separating/maxcutsroot": 1253325823, 
        "separating/minortho": 0.7668450596927363, 
        "separating/minorthoroot": 0.9071676731109619, 
        "separating/poolfreq": 48058}

def run_smac(instance_file, output_file):
    restore_incument = []
    for k, v in incumbent().items():
        restore_incument.append(f'{k}={v}')

    _, suggested_configs = optimize(instance_file, runcount_limit=5)

    with open(output_file, 'a') as f:
        for rank, config in enumerate(suggested_configs):
            params_str = json.dumps(str(config['params']))
            expected_cost = config['cost']
            cost = config['cost']
            time = config['time']
            line = f'smac;{rank};{expected_cost};{cost};{time};{params_str};None\n'
            f.write(line)


def run(params, instance, output_file, source, rank=1, expected_cost=None, distance=None):
    _, cost, time = solve_milp(params=params, instance=instance)
    with open(output_file, 'a') as f:
        params_str = json.dumps(str(params))
        line = f'{source};{rank};{expected_cost};{cost};{time};{params_str};{distance}\n'
        f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter, \
        description='Suggests configuration parameters for SCIP')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version', \
        version = f'MILPTune v{VERSION}', help='Shows program\'s version number and exit')
    parser.add_argument('instance', type=str, \
        help='Path to the instance.mps.gz file')
    parser.add_argument('dataset_name', type=str, \
        help='Dataset name')
    parser.add_argument('output_dir', type=str, \
        help='Specifies the output dir to append results to')
    args = parser.parse_args()
    
    output_file = os.path.join(args.output_dir, os.path.basename(args.instance).split('.')[0] + '.csv')
    
    # 1. Run default
    run(None, args.instance, output_file, 'default')

    # # 2. Run incumbent from SMAC
    # Process(target=run_smac, args=[args.instance, output_file]).start()

    # # 3. Run MILPTune configs
    # configs, distances = get_configuration_parameters(instance_file=args.instance, dataset_name=args.dataset_name, n_neighbors=5, n_configs=1)
    # for rank, config in enumerate(configs):
    #     run(config['params'], args.instance, output_file, 'milptune', rank, config['cost'], distances[rank])
