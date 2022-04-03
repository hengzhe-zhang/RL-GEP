from functools import partial

import gym
import numpy as np
from gym import spaces
from sklearn.metrics import mean_squared_error

import geppy as gep
from simple_utils import generate_primitive_set
from geppy.core.symbol import Terminal


class GEPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, X, Y, input_name, test_data=None, head_length=5, verbose=None, noisy_input=False):
        """
        """
        self.data = X, Y
        self.test_data = test_data
        self.train_error_list = []
        self.test_error_list = []
        self.train_gen_error_list = []
        self.test_gen_error_list = []
        self.source_list = []

        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
        # self.data = (x_train, x_test, y_train, y_test)
        self.id = 'ML'
        self.reward_threshold = 0
        self.trials = 1
        self.verbose = verbose

        self.pset = generate_primitive_set(input_name)
        action_len = len(self.pset.functions + self.pset.terminals)
        # GEP gene length
        self.head_length = head_length
        self.max_arity = 2
        gene_length = head_length + head_length * (self.max_arity - 1) + 1
        self.gene_length = gene_length

        self.action_len = action_len
        # action=primitives*gene length
        self.action_space = spaces.Discrete(action_len * gene_length)
        self.action_frequency = np.zeros((gene_length, action_len))
        # self.observation_space = spaces.Tuple((spaces.Discrete(action_len),) * gene_length)
        self.cache = {}
        self.train_data = self.data[0].flatten()
        self.best_ind = None
        self.current_step = 0
        self.noisy_input = noisy_input

    def best_score_record(self):
        best = self.best_ind
        X, y = self.data
        predicted_value, fun = self.gep_evaluate(X, best['action'], [], return_fun=True)
        predicted_value = np.concatenate([predicted_value.reshape(-1, 1), np.ones((len(predicted_value), 1))], axis=1)
        predicted_value = predicted_value.dot(best['coef'])
        score = -float(mean_squared_error(predicted_value, y))
        self.train_gen_error_list.append(-score)

        if self.test_data != None:
            X, y = self.test_data
            predicted_value = fun(*X.T)
            predicted_value = np.concatenate([predicted_value.reshape(-1, 1), np.ones((len(predicted_value), 1))],
                                             axis=1)
            predicted_value = predicted_value.dot(best['coef'])
            score = -float(mean_squared_error(predicted_value, y))
            self.test_gen_error_list.append(-score)

    def step(self, action):
        """
        This method is the primary interface between environment and agent.
        Paramters:
            action: int
                    the index of the respective action (if action space is discrete)
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        self.current_step += 1
        # if tuple(action) in self.cache:
        #     # return self.train_data, self.cache[tuple(action)], True
        #     return np.array([1]), self.cache[tuple(action)], True
        X, y = self.data
        predicted_value, fun = self.gep_evaluate(X, action, [], return_fun=True)
        predicted_value = np.concatenate([predicted_value.reshape(-1, 1), np.ones((len(predicted_value), 1))], axis=1)
        # linear scale
        coef, _, _, _ = np.linalg.lstsq(predicted_value, y.reshape(-1, 1), rcond=None)
        predicted_value = predicted_value.dot(coef)
        score = -float(mean_squared_error(predicted_value, y))
        self.train_error_list.append(-score)

        if self.test_data != None:
            X_test, y_test = self.test_data
            predicted_value = fun(*X_test.T)
            if type(predicted_value) in [int, float] or predicted_value.size == 1:
                predicted_value = np.full(X_test.shape[0], predicted_value)
            predicted_value = np.concatenate([predicted_value.reshape(-1, 1), np.ones((len(predicted_value), 1))],
                                             axis=1)
            predicted_value = predicted_value.dot(coef)
            score = -float(mean_squared_error(predicted_value, y_test))
            self.test_error_list.append(-score)

        for i, a in enumerate(action):
            self.action_frequency[i][a] += 1

        self.cache[tuple(action)] = score
        if self.best_ind is None or score > self.best_ind['score']:
            self.best_ind = {
                'action': action,
                'coef': coef,
                'score': score,
            }

        if self.current_step % 100 == 0:
            self.best_score_record()
        # return self.train_data, score, True
        if self.noisy_input:
            return np.random.randn(1), score, True
        else:
            return np.array([1]), score, True

    def reset(self):
        """
        This method resets the environment to its initial values.
        Returns:
            observation:    array
                            the initial state of the environment
        """
        # return self.train_data
        if self.noisy_input:
            return np.random.randn(1)
        else:
            return np.array([1])

    def reset_environment(self):
        return self.reset()

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        return

    def gep_evaluate(self, train_data, action, constant, return_fun=False):
        x = action
        if self.verbose:
            print(x, constant)

        gene = self.convert_action_to_gene(x)

        if type(gene[0][0]) == Terminal:
            return np.zeros(train_data.shape[0])
        if self.verbose:
            print('gene length', len(gene[0]), 'gene', gene, 'cache size', len(self.cache))
        fun = gep.compile_(gene, self.pset)
        predicted_value = fun(*train_data.T)
        if type(predicted_value) in [int, float] or predicted_value.size == 1:
            predicted_value = np.full(train_data.shape[0], predicted_value)
        if return_fun:
            return predicted_value, fun
        else:
            return predicted_value

    def convert_action_to_gene(self, x):
        pset = self.pset
        head_length = self.head_length
        gene_gen = partial(gep.Gene, pset, head_length)
        gene = gep.Chromosome(gene_gen, 1)
        nodes = (pset.functions + pset.terminals)
        # random padding
        for i in range(len(gene[0])):
            # gene[0][i] = Terminal(0, 0)
            if i < head_length:
                gene[0][i] = nodes[np.random.randint(0, len(nodes))]
            else:
                gene[0][i] = nodes[np.random.randint(len(pset.functions), len(nodes))]
        for i in range(len(x)):
            gene[0][i] = nodes[x[i]]
        return gene

    def convert_gene_to_action(self, x):
        pset = self.pset
        nodes = (pset.functions + pset.terminals)
        n: Terminal
        action = []
        remain_value = 1
        for a in x[0]:
            if remain_value <= 0:
                break
            for i, n in enumerate(nodes):
                if n.name == a.name:
                    action.append(i)
                    # function
                    if i < len(pset.functions):
                        remain_value += 2
                    break
            remain_value -= 1
        return action
