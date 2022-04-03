import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rl_algorithms.agents.Trainer import Trainer
from rl_algorithms.agents.policy_gradient_agents.REINFORCE import REINFORCE
from gep_gym import GEPEnv
from simple_utils import boston_house_data


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self, num_episodes_to_run: int, verbose: bool, entropy_loss: bool, hyperparameters: dict,
                 regularization_ratio: float, variable_length: bool, ga_probability: float, use_GPU: bool,
                 avoid_repetition: bool, adaptive_hybrid: bool, adaptive_fun: object, chromosome_fusion: bool):
        self.seed = 1
        self.environment: GEPEnv = None
        self.requirements_to_solve_game = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.visualise_overall_results = None
        self.debug_mode = False

        self.entropy_loss = entropy_loss
        self.num_episodes_to_run = num_episodes_to_run
        self.show_solution_score = False
        self.visualise_individual_results = False
        self.visualise_overall_agent_results = True
        self.standard_deviation_results = 1.0
        self.runs_per_agent = 1
        self.use_GPU = use_GPU
        self.overwrite_existing_results_file = False
        self.randomise_random_seed = True
        self.hyperparameters = hyperparameters
        self.verbose = verbose
        self.regularization_ratio = regularization_ratio
        self.variable_length = variable_length
        self.ga_probability = ga_probability
        self.chromosome_fusion = chromosome_fusion
        self.avoid_repetition = avoid_repetition
        self.adaptive_hybrid = adaptive_hybrid
        self.adaptive_fun = adaptive_fun


class RLGEPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_gen=20, verbose=False, head_length=10, entropy_loss=True, learning_rate=1e-3,
                 regularization_ratio=0.1, variable_length=False, ga_probability=0.0, use_GPU=False,
                 avoid_repetition=True, test_data=None, linear_hidden_units='[64, 64]', adaptive_hybrid=False,
                 noisy_input=False, adaptive_fun=None, chromosome_fusion=False, **param):
        # allow unused parameters
        self.head_length = head_length
        self.n_gen = n_gen
        self.test_data = test_data
        self.verbose = verbose
        hyperparameters = {"Policy_Gradient_Agents": {
            "learning_rate": learning_rate,
            "linear_hidden_units": linear_hidden_units,
            "discount_rate": 0.99,
            "clip_rewards": False
        }}
        self.noisy_input = noisy_input
        self.config = Config(num_episodes_to_run=n_gen, verbose=verbose, entropy_loss=entropy_loss,
                             hyperparameters=hyperparameters, regularization_ratio=regularization_ratio,
                             variable_length=variable_length, ga_probability=ga_probability, use_GPU=use_GPU,
                             avoid_repetition=avoid_repetition, adaptive_hybrid=adaptive_hybrid,
                             adaptive_fun=adaptive_fun, chromosome_fusion=chromosome_fusion)

    def fit(self, X, y):
        self.config.environment = GEPEnv(X, y, input_name=[f'X{i}' for i in range(X.shape[1])],
                                         head_length=self.head_length, test_data=self.test_data,
                                         noisy_input=self.noisy_input)
        env = self.config.environment
        env.verbose = self.verbose

        self.X, self.y = X, y
        trainer = Trainer(self.config, [REINFORCE])
        trainer.run_games_for_agents()

        """
        There are two kinds of lists:
        The first kind is a detailed list, which contains all fitness values of proposed individuals
        The second kind is a generational list, which record the best score after a few generations
        """
        self.train_error_list = self.config.environment.train_error_list
        self.test_error_list = self.config.environment.test_error_list
        self.action_frequency = self.config.environment.action_frequency
        self.train_gen_error_list = self.config.environment.train_gen_error_list
        self.test_gen_error_list = self.config.environment.test_gen_error_list
        self.source_list = self.config.environment.source_list
        return self

    def predict(self, X):
        env = self.config.environment
        best = env.best_ind
        predicted_value = env.gep_evaluate(X, best['action'], [])
        predicted_value = np.concatenate([predicted_value.reshape(-1, 1), np.ones((len(predicted_value), 1))], axis=1)
        predicted_value = predicted_value.dot(best['coef'])
        return predicted_value


# @notify
def simple_exp():
    # reset_random(0)
    X, y, input_name = boston_house_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
    reg = RLGEPRegressor(n_gen=500, verbose=True, head_length=15, learning_rate=1e-3,
                         variable_length=True, ga_probability=0.75, use_GPU=False,
                         test_data=(x_test, y_test), linear_hidden_units='[32,32]')
    reg.fit(x_train, y_train)
    print('Training Loss', mean_squared_error(y_train, reg.predict(x_train)))
    print('Testing Loss', mean_squared_error(y_test, reg.predict(x_test)))
    print('Training Curve', reg.train_error_list)
    print('Testing Curve', reg.test_error_list)
    print('Training Curve', reg.train_gen_error_list)
    print('Testing Curve', reg.test_gen_error_list)
    print('Action Frequency Count', reg.action_frequency)


if __name__ == '__main__':
    simple_exp()