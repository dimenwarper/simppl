import numpy as np
from itertools import product
from scipy.special import logsumexp
from collections import namedtuple
from .registry import REGISTRY

InferenceResults = namedtuple('InferenceResults', 'model_locals return_value')


def Enumerate(fun, **fun_kwargs):
    REGISTRY.reset(fun, **fun_kwargs)
    variables = REGISTRY.get_variables()

    all_supports = []
    all_results = {}
    Z = []
    for name, var in variables.items():
        all_supports.append([(name, supp) for supp in var.support_or_obs()])
    for defs in product(*all_supports):
        res, score = REGISTRY.call_with_definitions(fun, dict(defs))
        all_results.setdefault(res, []).append(score)
        Z.append(score)

    Z = logsumexp(Z)
    return InferenceResults(
        return_value=dict([(res, np.exp(logsumexp(scores) - Z)) for res, scores in all_results.items()]),
        model_locals=REGISTRY.model_locals
    )


def MCMC(fun, niter=100, **fun_kwargs):
    REGISTRY.reset(fun, **fun_kwargs)
    variables = REGISTRY.get_variables()

    trace = {vname: {} for vname in variables}

    prev_score = 0
    for it in range(niter):
        defs = {}
        for name, var in variables.items():
            defs[name] = np.random.choice(var.support)
        res, score = REGISTRY.call_with_defnitions(fun, defs, apply_observations=True)
        acceptance = min(1, np.exp(score - prev_score))
        if np.random.rand() > acceptance:
            for name, var in variables.items():
                if not var.is_observed():
                    trace[name].append(defs[name])
    return trace