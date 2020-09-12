from typing import List, Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import OneClassSVM

from . import viz
from .computation_registry import Variable


class Distribution(Variable):
    def __init__(self, name, observed=None, **kwargs):
        self.name = name
        self.observed = observed
        self.support = None
        self.log_probas = None

    def loglike(self, val):
        raise NotImplemented('Loglike method has to be implemented for random variables')

    def is_observed(self):
        return self.observed is not None

    def support_or_obs(self):
        if self.is_observed():
            return self.observed
        else:
            return self.support


############
# Custom distributions
############

class Flip(Distribution):
    def __init__(self, name, p=0.5, **kwargs):
        Distribution.__init__(self, name, **kwargs)
        self.p = p
        self.support = [0, 1]
        self.log_probas = {0: np.log(1 - p), 1: np.log(p)}

    def loglike(self, choice):
        return self.log_probas[choice]

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
        return self.log_probas[choice]

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
            **kwargs
    ):
        Distribution.__init__(self, name, **kwargs)
        self.N = 10
        self.between = between
        delta = (self.between[1] - self.between[0]) / self.N
        self.support = [between[0] + i * delta for i in range(self.N)]

        self.around = around if around is not None else []
        self.mostly = mostly

        self.log_probas = self.__generate_probas(self.support, self.around, self.mostly)

    def __get_best_supp_idx(self, x):
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
        # noinspection PyTypeChecker
        best_supp = self.support[self.__get_best_supp_idx(value)]
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
            [self.log_probas[s] for s in self.support],
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
        if 'float' in str(samples.dtype) or 'int' in str(samples.dtype):
            return pairwise_kernels(samples, metric=affinity_fun)
        else:
            N = len(samples)
            M = np.zeros([N, N])
            for i, x in enumerate(samples):
                for j in range(i, N):
                    y = samples[j]
                    M[i, j] = affinity_fun(x, y)
                    M[j, i] = M[i, j]
            return M

    @staticmethod
    def discretized_scores(resolution, samples, affinity_matrix, score_fun):
        clusterer = SpectralClustering(
            n_clusters=resolution,
            affinity='precomputed',
            assign_labels="discretize"
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
    def loglike(affinity_fun, log_probas, samples, sample):
        return log_probas[
            sorted([(idx, affinity_fun(sample, samples[idx])) for idx in log_probas],
                   key=lambda x: x[1]
                   )[-1][0]
        ]

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


class SomeSpace(Distribution):
    def __init__(
            self,
            name,
            samples: np.array,
            resolution: int = 10,
            affinity: str = 'rbf',
            **kwargs
    ):
        Distribution.__init__(self, name, **kwargs)
        self.samples = samples
        self.resolution = resolution
        self.affinity = affinity
        self.resolution = min(resolution, len(samples) // 2)
        self.__affinity_fun = KernelDiscretizedMethod.get_affinity_function(self.affinity)
        self.__affinity_matrix = KernelDiscretizedMethod.get_affinity_matrix(self.samples, self.__affinity_fun)

        self.log_probas = KernelDiscretizedMethod.discretized_scores(
            self.resolution,
            self.samples,
            self.__affinity_matrix,
            lambda mask, idx: mask.mean()
        )

        self.support = [self.samples[idx] for idx in self.log_probas]

    def loglike(self, sample):
        return KernelDiscretizedMethod.loglike(self.__affinity_fun, self.log_probas, self.samples, sample)

    def _repr_html_(self):
        return KernelDiscretizedMethod.render(self.__affinity_matrix, self.loglike, self.samples,
                                              f'SomeSpace {self.name}')


class AskTheExpert(Distribution):
    def __init__(
            self,
            name,
            samples,
            number_of_questions=10,
            affinity='rbf',
            resolution=100,
            **kwargs
    ):
        Distribution.__init__(name, **kwargs)
        self.samples = samples
        self.affinity = affinity
        self.resolution = min(resolution, len(samples) // 2)
        self.__affinity_fun = KernelDiscretizedMethod.get_affinity_function(self.affinity)
        self.__affinity_matrix = KernelDiscretizedMethod.get_affinity_matrix(self.samples, self.__affinity_fun)

        self.log_probas = self.__generate_probas(
            self.samples,
            self.resolution,
            self.__affinity_matrix,
            number_of_questions
        )

        self.support = [self.samples[idx] for idx in self.log_probas]

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

    def loglike(self, sample):
        return KernelDiscretizedMethod.loglike(self.__affinity_fun, self.log_probas, self.samples, sample)

    def _repr_html_(self):
        return KernelDiscretizedMethod.render(self.__affinity_matrix, self.loglike, self.samples,
                                              f'AskTheExpert {self.name}')
