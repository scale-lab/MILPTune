import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from milptune.scip.solver import solve_milp


def optimize(instance_file):
    # Configuration
    cs = ConfigurationSpace()
    params = [
        # Branching
        CategoricalHyperparameter('branching/scorefunc', choices=['s', 'p', 'q'], default_value='p'),
        UniformFloatHyperparameter('branching/scorefac', 0.0, 1.0, default_value=0.167),
        CategoricalHyperparameter('branching/preferbinary', choices=[True, False], default_value=False),
        UniformFloatHyperparameter('branching/clamp', 0.0, 0.5, default_value=0.2),
        UniformFloatHyperparameter('branching/midpull', 0.0, 1.0, default_value=0.75),
        UniformFloatHyperparameter('branching/midpullreldomtrig', 0.0, 1.0, default_value=0.5),
        CategoricalHyperparameter('branching/lpgainnormalize', choices=['d', 'l', 's'], default_value='s'),
        # LP
        CategoricalHyperparameter('lp/pricing', choices=['l','a','f','p','s','q','d'], default_value='l'),
        UniformIntegerHyperparameter('lp/colagelimit', -1, 2147483647, default_value=10),
        UniformIntegerHyperparameter('lp/rowagelimit', -1, 2147483647, default_value=10),
        # Node Selection
        CategoricalHyperparameter('nodeselection/childsel', choices=["d",'u','p','i','l','r','h'], default_value='h'),
        # Separating
        UniformFloatHyperparameter('separating/minortho', 0.0, 1.0, default_value=0.9),
        UniformFloatHyperparameter('separating/minorthoroot', 0.0, 1.0, default_value=0.9),
        UniformIntegerHyperparameter('separating/maxcuts', 0, 2147483647, default_value=100),
        UniformIntegerHyperparameter('separating/maxcutsroot', 0, 2147483647, default_value=2000),
        UniformIntegerHyperparameter('separating/cutagelimit', -1, 2147483647, default_value=80),
        UniformIntegerHyperparameter('separating/poolfreq', -1, 65534, default_value=10)
    ]
    cs.add_hyperparameters(params)

    # Scenario object
    scenario = Scenario(
        {
            'run_obj': 'quality',       # we optimize quality (alternatively runtime)
            'runcount-limit': 8,        # max. number of function evaluations
            'cs': cs,                   # configuration space
            'deterministic': True,
            'instances': [[instance_file]],
        }
    )

    seed = np.random.randint(1000000, 9999999)
    smac = SMAC4HPO(
        scenario=scenario, tae_runner=solve_milp,
        rng=np.random.RandomState(seed), run_id=seed)
    
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    configs = []
    for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in smac.runhistory.data.items():
        config = {
            'seed': seed,
            'cost': cost,
            'time': time,
            'params': smac.runhistory.ids_config[config_id]._values
        }
        configs.append(config)

    return incumbent, configs
