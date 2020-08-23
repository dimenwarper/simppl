import numpy as np
from typing import List, Union, Optional
from .registry import REGISTRY

class Distribution:
    def __new__(cls, name, *args, register=True, **kwargs):
        return REGISTRY.call_variable(name, cls, register, *args, **kwargs)

    def __init__(self, register=False, observed=None):
        self.observed = observed
        self.support = None

    def loglike(self, val):
        raise NotImplemented('Loglike method has to be implemented for random variables')

    def score(self, val):
        if self.observed is None:
            return self.loglike(val)
        else:
            return sum([self.loglike(obs) for obs in self.observed])

    def is_observed(self):
        return self.observed is not None

    def support_or_obs(self):
        if self.is_observed():
            return self.observed
        else:
            return self.support


class Flip(Distribution):
    def __init__(self, name, p=0.5, **kwargs):
        Distribution.__init__(self, **kwargs)
        self.p = p
        self.support = [0, 1]

    def loglike(self, choice):
        return np.log(1 - self.p if choice == 0 else self.p)

    def __repr__(self):
        return f'Flip(name={self.name} p={self.p})'


class Pick(Distribution):
    def __init__(self, name, items=[], **kwargs):
        Distribution.__init__(self, **kwargs)
        self.items = items
        self.support = items

    def loglike(self, choice):
        return np.log(1 / len(self.support))

    def __repr__(self):
        return f'Pick(name={self.name} items={self.items})'


class SomeValue(Distribution):
    def __init__(
            self,
            name,
            between,
            around: Optional[List[Union[float, np.array]]] = None,
            mostly: Optional[Union[float, np.array]] = None,
            **kwargs
    ):
        Distribution.__init__(self, **kwargs)
        self.N = 10
        self.between = between
        delta = (self.between[1] - self.between[0]) / self.N
        self.support = [between[0] + i * delta for i in range(self.N)]

        self.around = around if around is not None else []
        self.mostly = mostly

        self.__generate_probas()

    def __get_best_supp_idx(self, x):
        return np.argmin([np.linalg.norm(s - x) for s in self.support])

    def __add_factor(self, pivot, factor, decay):
        best_supp_idx = self.__get_best_supp_idx(pivot)
        for i in range(self.N):
            if best_supp_idx + i < self.N:
                self.probas[self.support[best_supp_idx + i]] += decay(factor, i + 1)
            if best_supp_idx - i >= 0:
                self.probas[self.support[best_supp_idx - i]] += decay(factor, i + 1)

    def __generate_probas(self):
        self.probas = {s: 1 for s in self.support}
        for mode in self.around:
            self.__add_factor(pivot=mode, factor=10, decay=lambda factor, distance: factor / distance)

        if self.mostly is not None:
            self.__add_factor(pivot=self.mostly, factor=50, decay=lambda factor, distance: factor / (distance ** 2))

        Z = sum(list(self.probas.values()))
        self.probas = {k: p / Z for k, p in self.probas.items()}

    def loglike(self, value):
        best_supp = self.support[self.__get_best_supp_idx(value)]
        return np.log(self.probas[best_supp])

    def __repr__(self):
        r = f'༺ SomeValue "{self.name}"\n\t BETWEEN {self.between[0]} AND {self.between[1]}'
        if self.around:
            r += f' \n\t AROUND {self.around}'
        if self.mostly:
            r += f' \n\t BUT MOSTLY {self.mostly}'
        r += ' ༻'
        return r
