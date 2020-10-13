from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from itertools import product
from scipy.special import logsumexp
from tqdm.auto import tqdm

from .computation_registry import COMPUTATION_REGISTRY, CNode, Op
from .distributions import Distribution


class Model:
    def __init__(self, distributions: Dict[str, Distribution]):
        self.distributions = distributions

    def _is_in_model_tree(self, d: Distribution, model_tree: Dict[str, Dict[str, Any]]) -> bool:
        if len([dd for dd in model_tree.values() if dd['distribution'].name == d.name]) == 0:
            for node in model_tree.values():
                if self._is_in_model_tree(d, node['children']):
                    return True
            return False
        else:
            return True

    # TODO: this should really be in computation_registry
    def __extract_distributions(self, val, carried=[]):
        if isinstance(val, CNode):
            if isinstance(val, Distribution):
                return [val]
            elif isinstance(val, Op):
                for arg in val.args:
                    carried += self.__extract_distributions(arg, carried)
                return carried
            else:
                return []
        else:
            return []

    def __build_model_tree(self, distributions: Dict[str, Distribution]) -> Dict[str, Dict[str, Any]]:
        if len(distributions) == 0:
            return {}

        model_tree = {}
        remaining = []
        for name, d in distributions.items():
            dist_children = {k: child_dist for k, v in d.__dict__.items() for child_dist in self.__extract_distributions(v)}
            if len(dist_children) > 0:
                model_tree[name] = {
                    'distribution': d,
                    'children': self.__build_model_tree(dist_children)
                }
            else:
                remaining.append((name, d))

        for name, d in remaining:
            if not self._is_in_model_tree(d, model_tree):
                model_tree[name] = {
                    'distribution': d,
                    'children' : {}
                }

        return model_tree

    @property
    def model_tree(self):
        return self.__build_model_tree(self.distributions)

    def __render_model_tree(self, model_tree: Dict[str, Dict[str, Any]], level: Optional[int] = 0, prefix: Optional[str] = '   '):
        html = ''
        N = len(model_tree)
        for i, (name, node) in enumerate(model_tree.items()):
            if i == 0 and level == 0 and i != N - 1:
                gylph = '┌'
            elif i == N - 1:
                gylph = '└'
            else:
                gylph = '├'
            next_prefix = '  '
            if i < N - 1:
                next_prefix = '│  '
            if len(node['children']) == 0:
                html += prefix + f'{gylph}──⮀ {name} = {node["distribution"].__repr__()}\n'
            else:
                html += prefix + f'{gylph}──⮀ {node["distribution"].__class__.__name__} name={node["distribution"].name}\n'
                html += f'{self.__render_model_tree(node["children"], level + 1, prefix + next_prefix)}'

        return html

    def __render_distributions(self, model_tree: Dict[str, Dict[str, Any]], html: Optional[str] = '', already_rendered: Optional[List[str]] = None):
        _already_rendered = already_rendered if already_rendered is not None else []
        for name, node in model_tree.items():
            if len(node['children']) == 0 and node['distribution'] not in _already_rendered:
                _already_rendered.append(node['distribution'].name)
                html += f'<span>{node["distribution"]._repr_html_()}</span>'
            else:
                html += self.__render_distributions(node['children'], already_rendered=_already_rendered)
        return html

    def _repr_html_(self):
        html = '<div>'
        html += '<h4>Model Tree</h4>'
        html += f'<div><pre>{self.__render_model_tree(self.model_tree, 0)}</pre></div>'
        html += '<h4>Priors</h4>'
        html += f'<div>{self.__render_distributions(self.model_tree)}</div>'
        html += '</div>'
        return html


class RandomComputationEnv:
    def __init__(self, model, model_locals, return_values, normalization_constant):
        self.model = model
        self.__model_locals = model_locals
        self.__model_locals['_probability_'] = [
            np.exp(logsumexp(v) - normalization_constant) if len(v) > 0 else 0
            for v in self.__model_locals['_scores_']
        ]
        self.executions = self.__build_executions_df(self.__model_locals)
        self.return_values = return_values

    def __build_executions_df(self, locals: Dict[Any, Any]) -> pd.DataFrame:
        if len(locals) == 0:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(locals).drop('_scores_', axis=1)

    def _repr_html_(self):
        r = '👋 You are looking at a <i>Random Computation Environment</i>, the <b>executions</b> property has a dataframe' \
            ' with all of the executions visited during inference, while the <b>model</b> property has a structured' \
            ' version of the model used.'
        return r




def Exhaustive(fun, **fun_kwargs):
    COMPUTATION_REGISTRY.reset(fun, **fun_kwargs)
    variables = COMPUTATION_REGISTRY.current_variables

    all_supports = []
    all_results = {}
    Z = []
    for name, var in variables.items():
        all_supports.append([(name, supp) for supp in var.support])

    for defs in tqdm(list(product(*all_supports))):
        defs = dict(defs)
        res, realized_variables = COMPUTATION_REGISTRY.call_with_definitions(fun, defs)

        scores = []
        for name, v in realized_variables.items():
            if isinstance(v, Distribution):
                if not v.is_observed():
                    scores += [v.loglike(defs[name])]

        for name, v in realized_variables.items():
            if isinstance(v, Distribution):
                if v.is_observed():
                    scores += [v.loglike(v.observations[i, :]) for i in range(len(v.observations))]

        COMPUTATION_REGISTRY.add_to_model_locals(**{'_return_value_': res, '_scores_': scores})

        all_results.setdefault(res, []).extend(scores)
        Z.append(logsumexp(scores))

    Z = logsumexp(Z)
    return RandomComputationEnv(
        model=Model(variables),
        return_values=dict(
            [
                (res, np.exp(logsumexp(scores) - Z) if len(scores) > 0 else 0)
                for res, scores in all_results.items() if len(scores) > 0
            ]
        ),
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