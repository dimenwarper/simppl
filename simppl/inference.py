from types import SimpleNamespace

import numpy as np
from itertools import product
from scipy.special import logsumexp
from .computation_registry import COMPUTATION_REGISTRY

class RandCompEnv:
    def __init__(self, model_locals, return_value, normalization_constant=None):
        self.__model_locals = model_locals
        if normalization_constant is not None:
            for vals in self.__model_locals.values():
                for v in vals:
                    v[-1] = np.exp(v[-1]) - normalization_constant
        self.locals = SimpleNamespace(**self.__model_locals)
        self.return_value = return_value

def Enumerate(fun, **fun_kwargs):
    COMPUTATION_REGISTRY.reset(fun, **fun_kwargs)
    variables = COMPUTATION_REGISTRY.current_variables

    all_supports = []
    all_results = {}
    Z = []
    for name, var in variables.items():
        all_supports.append([(name, supp) for supp in var.support_or_obs()])
    for defs in product(*all_supports):
        res, score = COMPUTATION_REGISTRY.call_with_definitions(fun, dict(defs))
        all_results.setdefault(res, []).append(score)
        Z.append(score)
    Z = logsumexp(Z)
    return RandCompEnv(
        return_value=dict([(res, np.exp(logsumexp(scores) - Z)) for res, scores in all_results.items()]),
        model_locals=COMPUTATION_REGISTRY.model_locals,
        normalization_constant=Z
    )


def MCMC(fun, niter=100, **fun_kwargs):
    COMPUTATION_REGISTRY.reset(fun, **fun_kwargs)
    variables = COMPUTATION_REGISTRY.current_variables

    trace = {vname: {} for vname in variables}

    prev_score = 0
    for it in range(niter):
        defs = {}
        for name, var in variables.items():
            defs[name] = np.random.choice(var.support)
        res, score = COMPUTATION_REGISTRY.call_with_defnitions(fun, defs, apply_observations=True)
        acceptance = min(1, np.exp(score - prev_score))
        if np.random.rand() > acceptance:
            for name, var in variables.items():
                if not var.is_observed():
                    trace[name].append(defs[name])
    return trace