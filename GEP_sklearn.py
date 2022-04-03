import datetime
import random
import time

import numpy as np
from deap import creator, base, tools
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import geppy as gep
from simple_utils import boston_house_data, generate_primitive_set


class GEPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_pop=50, n_gen=20, verbose=False, head_length=10,**param):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.verbose = verbose
        self.n_genes = 1
        self.head_length = head_length

    def evaluate_ls(self, individual):
        """
        First apply linear scaling (ls) to the individual
        and then evaluate its fitness: MSE (mean squared error)
        """
        X, Y = self.X, self.y

        func = self.toolbox.compile(individual)
        Yp = np.array(list(map(func, *X.T)))

        if isinstance(Yp, np.ndarray):
            Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
            (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)

            if residuals.size > 0:
                return residuals[0] / len(Y),

        individual.a = 0
        individual.b = np.mean(Y)
        return np.mean((Y - individual.b) ** 2),

    def lazy_init(self, x):
        pset = generate_primitive_set([f'X{i}' for i in range(x.shape[1])])

        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

        toolbox = gep.Toolbox()
        self.toolbox = toolbox
        toolbox.register('rnc_gen', random.randint, a=-10, b=10)
        toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=self.head_length)
        toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=self.n_genes)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register('compile', gep.compile_, pset=pset)

        toolbox.register('evaluate', self.evaluate_ls)

        toolbox.register('select', tools.selTournament, tournsize=5)

        toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
        toolbox.register('mut_invert', gep.invert, pb=0.1)
        toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
        toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
        toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
        toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
        toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
        toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        self.stats = stats
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.pop = toolbox.population(n=self.n_pop)
        self.hof = tools.HallOfFame(1)

        startDT = datetime.datetime.now()
        print(str(startDT))

    def fit(self, X, y):
        self.X, self.y = X, y
        self.lazy_init(X)
        pop, log = gep.gep_simple(self.pop, self.toolbox, n_generations=self.n_gen, n_elites=10,
                                  stats=self.stats, hall_of_fame=self.hof, verbose=self.verbose,)
        return self

    def predict(self, X):
        individual = self.hof[0]
        func = self.toolbox.compile(individual)
        Yp = np.array(list(map(func, *X.T)))

        if isinstance(Yp, np.ndarray):
            Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
            return Q @ np.array([individual.a, individual.b])
        return np.zeros(len(X))


s = 0
random.seed(s)
np.random.seed(s)

threshold = 1e-6


def protect_divide(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > threshold, np.divide(x1, x2), 1.)


if __name__ == '__main__':
    st = time.time()
    X, y, input_name = boston_house_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
    reg = GEPRegressor(n_gen=50, n_pop=50, head_length=20, verbose=True)
    reg.fit(x_train, y_train)
    print(mean_squared_error(y_train, reg.predict(x_train)))
    print(mean_squared_error(y_test, reg.predict(x_test)))
    print('time', time.time() - st)
