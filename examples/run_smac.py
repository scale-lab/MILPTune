import argparse

import numpy as np
import pyscipopt
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from pymongo import MongoClient
from pyscipopt import Model as SCIPModel
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from milptune.version import VERSION


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = "Usage: "
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


def solve_milp(params, instance):
    params = {k: params[k] for k in params}

    model = SCIPModel()
    model.readProblem(instance)
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParams(params)
    model.setParam("limits/time", 15 * 60)
    model.hideOutput()

    model.optimize()

    # solution
    _ = model.getBestSol()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    _ = model.getSolvingTime()

    return primal - dual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=CapitalisedHelpFormatter,
        description="Runs SMAC on the given instance",
    )
    parser._positionals.title = "Positional arguments"
    parser._optionals.title = "Optional arguments"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"MILPTune v{VERSION}",
        help="Shows program's version number and exit",
    )
    parser.add_argument(
        "dataset_name", type=str, help="Dataset name in the DB to add these runs to"
    )
    args = parser.parse_args()

    # Get instance from the DB
    uri = "mongodb://%s:%s@%s:%s" % ("milptune", "MILP*tune2023", "20.25.127.142", 31331)
    client = MongoClient(uri)
    db = client.milptunedb
    dataset = db[args.dataset_name]
    doc = dataset.find_one_and_update(
        {"configs": {"$exists": False}, "solver_lock": {"$exists": False}},
        {"$set": {"solver_lock": True}},
    )
    instance_file = doc["path"]

    # Configuration
    cs = ConfigurationSpace()
    params = [
        # Branching
        CategoricalHyperparameter(
            "branching/scorefunc", choices=["s", "p", "q"], default_value="p"
        ),
        UniformFloatHyperparameter("branching/scorefac", 0.0, 1.0, default_value=0.167),
        CategoricalHyperparameter(
            "branching/preferbinary", choices=[True, False], default_value=False
        ),
        UniformFloatHyperparameter("branching/clamp", 0.0, 0.5, default_value=0.2),
        UniformFloatHyperparameter("branching/midpull", 0.0, 1.0, default_value=0.75),
        UniformFloatHyperparameter("branching/midpullreldomtrig", 0.0, 1.0, default_value=0.5),
        CategoricalHyperparameter(
            "branching/lpgainnormalize", choices=["d", "l", "s"], default_value="s"
        ),
        # LP
        CategoricalHyperparameter(
            "lp/pricing", choices=["l", "a", "f", "p", "s", "q", "d"], default_value="l"
        ),
        UniformIntegerHyperparameter("lp/colagelimit", -1, 2147483647, default_value=10),
        UniformIntegerHyperparameter("lp/rowagelimit", -1, 2147483647, default_value=10),
        # Node Selection
        CategoricalHyperparameter(
            "nodeselection/childsel", choices=["d", "u", "p", "i", "l", "r", "h"], default_value="h"
        ),  # noqa
        # Separating
        UniformFloatHyperparameter("separating/minortho", 0.0, 1.0, default_value=0.9),
        UniformFloatHyperparameter("separating/minorthoroot", 0.0, 1.0, default_value=0.9),
        UniformIntegerHyperparameter("separating/maxcuts", 0, 2147483647, default_value=100),
        UniformIntegerHyperparameter("separating/maxcutsroot", 0, 2147483647, default_value=2000),
        UniformIntegerHyperparameter("separating/cutagelimit", -1, 2147483647, default_value=80),
        UniformIntegerHyperparameter("separating/poolfreq", -1, 65534, default_value=10),
    ]
    cs.add_hyperparameters(params)

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 8,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
            "instances": [[instance_file]],
        }
    )

    seed = np.random.randint(1000000, 9999999)
    smac = SMAC4HPO(
        scenario=scenario, tae_runner=solve_milp, rng=np.random.RandomState(seed), run_id=seed
    )

    try:
        print(instance_file)
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    for (config_id, instance_id, seed, budget), (
        cost,
        time,
        status,
        starttime,
        endtime,
        additional_info,
    ) in smac.runhistory.data.items():  # noqa
        config = {
            "seed": seed,
            "cost": cost,
            "time": time,
            "params": smac.runhistory.ids_config[config_id]._values,
        }
        r = dataset.find_one_and_update({"path": instance_id}, {"$push": {"configs": config}})

        print(r["_id"])
