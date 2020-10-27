from typing import List, Union, Optional, Dict, Any, Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import OneClassSVM

from . import viz
from .computation_registry import Variable, CNode, COMPUTATION_REGISTRY


class Distribution(Variable):
    def __init__(self, name, observations=None, **kwargs):
        self.name = name
        self.observations = observations
        self.support = None
        self.log_probas = None

    def loglike(self, val):
        raise NotImplemented('Loglike method has to be implemented for random variables')

    def is_observed(self):
        return self.observations is not None

    def support_or_obs(self):
        if self.is_observed():
            return self.observations
        else:
            return self.support


############
# Custom distributions
############

def _asscalar(x: Union[float, np.array]) -> bool:
    if isinstance(x, np.ndarray):
        return np.asscalar(x)
    return x

class Flip(Distribution):
    def __init__(self, name, p=0.5, **kwargs):
        Distribution.__init__(self, name, **kwargs)
        self.p = p
        self.support = [0, 1]
        self.log_probas = {0: np.log(1 - p), 1: np.log(p)}

    def loglike(self, choice):
        return self.log_probas[_asscalar(choice)]

    def __repr__(self):
        return f'Flip(name={self.name} p={self.p})'

    def _repr_html_(self):
        return viz.fracbar(
            self.support,
            {0: 1 - self.p, 1: self.p},
            colors=['lightgray', 'black'],
            label_colors=['black', 'white'],
            title=f'Flip {self.name}'
        )


class Pick(Distribution):
    def __init__(self, name, items=None, probas=None, **kwargs):
        Distribution.__init__(self, name, **kwargs)
        self.items = [] if items is None else items
        self.support = items
        if probas is None:
            self.log_probas = {s: np.log(1 / len(self.support)) for s in self.support}
        else:
            self.log_probas = {s: np.log(p) for s, p in probas.items()}

    def loglike(self, choice):
        return self.log_probas[_asscalar(choice)]

    def __repr__(self):
        return f'Pick(name={self.name} items={self.items})'

    def _repr_html_(self):
        sp = sorted(self.log_probas.items(), key=lambda x: x[1], reverse=True)
        return viz.sparkbars(
            range(len(self.support)),
            [v for k, v in sp],
            title=f'Pick {self.name}',
            xticklabels=[k for k, v in sp]
        )


class SomeValue(Distribution):
    def __init__(
            self,
            name,
            between,
            around: Optional[List[Union[float, np.array]]] = None,
            mostly: Optional[Union[float, np.array]] = None,
            resolution: Optional[int] = None,
            **kwargs
    ):
        Distribution.__init__(self, name, **kwargs)
        if COMPUTATION_REGISTRY.resolution is not None and resolution is None:
            self.N = COMPUTATION_REGISTRY.resolution
        else:
            self.N = resolution if resolution is not None else 10
        self.between = between
        delta = (self.between[1] - self.between[0]) / self.N
        self.support = [between[0] + i * delta for i in range(self.N)]

        self.around = around if around is not None else []
        self.mostly = mostly

        self.log_probas = self.__generate_probas(self.support, self.around, self.mostly)

    def __get_best_supp_idx(self, x):
        assert np.isscalar(x) or isinstance(x, CNode), 'SomeValue accepts only scalars'
        return np.argmin([np.linalg.norm(s - x) for s in self.support])

    def __add_factor(self, support, probas, pivot, factor, decay):
        best_supp_idx = self.__get_best_supp_idx(pivot)
        for i in range(self.N):
            if best_supp_idx + i < self.N:
                probas[support[best_supp_idx + i]] += decay(factor, i + 1)
            if best_supp_idx - i >= 0:
                probas[support[best_supp_idx - i]] += decay(factor, i + 1)
        return probas

    def __generate_probas(self, support, around, mostly):
        probas = {s: 1 for s in support}
        for mode in around:
            probas = self.__add_factor(support, probas, pivot=mode, factor=10,
                                       decay=lambda factor, distance: factor / distance)

        if mostly is not None:
            probas = self.__add_factor(support, probas, pivot=mostly, factor=50,
                                       decay=lambda factor, distance: factor / (distance ** 2))

        Z = sum(list(probas.values()))
        return {k: np.log(p / Z) for k, p in probas.items()}

    def loglike(self, value):
        val = np.asscalar(value) if isinstance(value, np.array([]).__class__) else value
        best_supp = self.support[self.__get_best_supp_idx(val)]
        return self.log_probas[best_supp]

    def __repr__(self):
        r = f'SomeValue( "{self.name}" BETWEEN {self.between[0]} AND {self.between[1]}'
        if self.around:
            r += f' AROUND {self.around}'
        if self.mostly:
            r += f' BUT MOSTLY {self.mostly}'
        return r + ')'

    def _repr_html_(self):
        return viz.sparkbars(
            self.support,
            [np.exp(self.log_probas[s]) for s in self.support],
            title=f'SomeValue {self.name}',
            width=(self.between[1] - self.between[0]) / len(self.support)
        )


class KernelDiscretizedMethod:
    @staticmethod
    def get_affinity_function(affinity):
        if type(affinity) == str:
            return lambda x, y: pairwise_kernels(
                x.reshape(1, -1) if len(x.shape) == 1 else x,
                y.reshape(1, -1) if len(y.shape) == 1 else y,
                metric=affinity
            )
        else:
            return affinity

    @staticmethod
    def get_affinity_matrix(samples, affinity_fun):
        N = len(samples)
        M = np.zeros([N, N])
        for i in range(N):
            x = samples[i, :]
            for j in range(i, N):
                y = samples[j, :]
                M[i, j] = affinity_fun(x, y)
                M[j, i] = M[i, j]
        return M

    @staticmethod
    def discretized_scores(resolution, samples, affinity_matrix, score_fun):
        clusterer = SpectralClustering(
            n_clusters=resolution,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=1234
        )

        labels = clusterer.fit_predict(affinity_matrix)

        scores = {}
        for i in np.unique(labels):
            mask = labels == i
            idx = np.argmax(affinity_matrix[mask, :][:, mask].sum(axis=1))
            idx = np.arange(len(samples))[mask][idx]
            scores[idx] = score_fun(mask, idx)
        return scores

    @staticmethod
    def get_best_affinity_idx_for_sample(affinity_fun, log_probas, sample, samples):
        centroid_indices = list(log_probas.keys())
        affs = affinity_fun(sample, samples[centroid_indices])
        return centroid_indices[np.argmin(affs)]

    @staticmethod
    def loglike(affinity_fun, log_probas, samples, sample):
        return log_probas[KernelDiscretizedMethod.get_best_affinity_idx_for_sample(affinity_fun, log_probas, sample, samples)]

    @staticmethod
    def render(affinity_matrix, log_fun, samples, title):
        fig = plt.figure(figsize=(2, 2))
        tx = SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(affinity_matrix)
        plt.scatter(
            tx[:, 0],
            tx[:, 1],
            s=30,
            alpha=0.4,
            c=[
                log_fun(samples[i])
                for i in range(samples.shape[0])
            ],
            cmap=plt.get_cmap('plasma')
        )
        plt.title(title)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        return viz.fig2html(fig)


class SomeThing(Distribution):
    def __init__(
            self,
            name,
            samples: np.array,
            mostly: Optional[np.array] = None,
            resolution: int = 10,
            affinity: str = 'rbf',
            affinity_matrix: Optional[np.array] = None,
            log_probas: Optional[Dict[Any, float]] = None,
            **kwargs
    ):
        Distribution.__init__(self, name, **kwargs)
        self.samples = samples
        self.resolution = resolution
        self.affinity = affinity
        self.resolution = min(resolution, len(samples) // 2)
        self.__affinity_fun = KernelDiscretizedMethod.get_affinity_function(self.affinity)
        self.__affinity_matrix = affinity_matrix if affinity_matrix is not None else KernelDiscretizedMethod.get_affinity_matrix(self.samples, self.__affinity_fun)

        if log_probas is None:
            self.log_probas = KernelDiscretizedMethod.discretized_scores(
                self.resolution,
                self.samples,
                self.__affinity_matrix,
                lambda mask, idx: mask.mean()
            )
        else:
            self.log_probas = log_probas

        self.mostly = mostly
        if self.mostly is not None and not isinstance(self.mostly, CNode):
            self.log_probas = self.__add_factor(
                lambda d: 10000 * d,
                self.__affinity_fun,
                self.log_probas,
                self.mostly,
                self.samples
            )

        self.support = [self.samples[idx, :] for idx in self.log_probas]

    def clone(self, *args, **kwargs):
        kwargs.update({'log_probas': self.log_probas, 'affinity_matrix': self.__affinity_matrix})
        return SomeThing(*args, **kwargs)

    def loglike(self, sample):
        return KernelDiscretizedMethod.loglike(self.__affinity_fun, self.log_probas, self.samples, sample)

    def __add_factor(self, factor_fun: Callable, affinity_fun, log_probas: Dict[Any, float], pivot: np.array, samples: np.array) -> Dict[Any, float]:
        pivot_idx = KernelDiscretizedMethod.get_best_affinity_idx_for_sample(affinity_fun, log_probas, pivot, samples)
        factored = {}
        aff_to_pivot = affinity_fun(samples[list(log_probas.keys()), :], samples[pivot_idx, :])
        factors = factor_fun(aff_to_pivot)
        for i, (idx, lp) in enumerate(log_probas.items()):
            factored[idx] = lp + np.asscalar(factors[i])
        Z = logsumexp(list(factored.values()))
        return {k: v - Z for k, v in factored.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}(resolution={self.resolution}, samples=[{"X".join([str(x) for x in self.samples.shape])}], affinity={self.__affinity_fun})'

    def _repr_html_(self):
        return KernelDiscretizedMethod.render(self.__affinity_matrix, self.loglike, self.samples,
                                              f'{self.__class__.__name__}{self.name}')


class AskTheExpert(SomeThing):
    def __init__(
            self,
            name,
            samples,
            mostly: Optional[np.array]=None,
            number_of_questions=10,
            affinity='rbf',
            resolution=100,
            **kwargs
    ):
        __affinity_fun = KernelDiscretizedMethod.get_affinity_function(affinity)
        __affinity_matrix = KernelDiscretizedMethod.get_affinity_matrix(samples, __affinity_fun)

        log_probas = self.__generate_probas(
            samples,
            resolution,

            number_of_questions
        )

        SomeThing.__init__(
                self,
                name=name,
                samples=samples,
                mostly=mostly,
                resolution=resolution,
                affinity=affinity,
                affinity_matrix=__affinity_matrix,
                log_probas=log_probas,
                **kwargs
        )

    def __generate_probas(self, samples, resolution, affinity_matrix, number_of_questions):
        print(
            f"📞 Looks like there's a probability distribution ({self.name}) that wants to phone in an expert (that's "
            f"you)\n"
        )
        clf = OneClassSVM(kernel='precomputed')
        samples_and_weights = {0: 0.5}
        for nq in range(number_of_questions):
            indices = list(samples_and_weights.keys())
            if nq == 0:
                idx = np.random.choice(range(1, len(samples)))
            else:
                preds = clf.decision_function(affinity_matrix[:, indices])
                idx = [i for i, _ in sorted(enumerate(preds), key=lambda x: x[1]) if i not in samples_and_weights][
                    0]
            sample = samples[idx]

            print('Score the sample below with a number between 0 and 1 (higher is better)\n')
            if hasattr(sample, '_repr_html_'):
                print(sample)
            else:
                print(sample)
            weight = float(input('Score: '))
            assert 0 <= weight <= 1

            samples_and_weights[idx] = weight
            indices = list(samples_and_weights.keys())
            clf.fit(
                affinity_matrix[indices, :][:, indices],
                sample_weight=list(samples_and_weights.values())
            )

        indices = list(samples_and_weights.keys())
        preds = clf.decision_function(affinity_matrix[:, indices])
        scores = KernelDiscretizedMethod.discretized_scores(
            resolution,
            samples,
            affinity_matrix,
            lambda mask, _idx: preds[mask].mean())

        Z = logsumexp([s for s in scores.values()])

        return {idx: s - Z for idx, s in scores.items()}

