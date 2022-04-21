import pyscipopt

from pyscipopt import Model as SCIPModel


def solve_milp(params=None, instance=''):
    model = SCIPModel()
    model.readProblem(instance)
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam('limits/time', 15 * 60)
    if params:
        params = {k: params[k] for k in params}
        model.setParams(params)
    model.hideOutput()

    model.optimize()

    # solution
    sol = model.getBestSol()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    time = model.getSolvingTime()

    return sol, primal - dual, time
