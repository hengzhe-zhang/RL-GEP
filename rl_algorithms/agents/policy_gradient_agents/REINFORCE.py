import random
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from deap.base import Fitness
from deap.tools import HallOfFame, selTournament
from torch.distributions import Categorical

import geppy as gep
from rl_algorithms.agents.Base_Agent import Base_Agent
from geppy import Gene, Chromosome
from geppy.algorithms.basic import _apply_modification, _apply_crossover


class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["learning_rate"])
        self.episode_rewards = []
        self.history_rewards = []
        self.episode_log_probabilities = []
        self.batch_size = 1
        self.history_action = set()
        self.entropy_loss = config.entropy_loss
        self.regularization_ratio = config.regularization_ratio
        self.variable_length = config.variable_length
        self.ga_probability = config.ga_probability
        self.chromosome_fusion = config.chromosome_fusion
        self.initial_ga_probability = self.ga_probability
        self.adaptive_hybrid = config.adaptive_hybrid
        self.adaptive_fun = config.adaptive_fun
        # the pop size is equivalent to 100
        self.hall_of_fame = HallOfFame(10)
        Fitness.weights = (-1,)
        genes = [Chromosome(partial(Gene, self.environment.pset, self.environment.head_length), 1) for _ in range(90)]
        for g in genes:
            g.fitness = Fitness((1e9,))
        self.pop = deque(genes)
        self.toolbox_initialization()
        self.generated_by_ga = False

    def toolbox_initialization(self):
        toolbox = gep.Toolbox()
        toolbox.register('mut_uniform', gep.mutate_uniform, pset=self.environment.pset, ind_pb=0.05, pb=1)
        toolbox.register('mut_invert', gep.invert, pb=0.1)
        toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
        toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
        toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
        toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
        toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
        toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
        self.toolbox = toolbox

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset_environment()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        for i in range(self.batch_size):
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.store_reward()
            assert len(self.episode_rewards) == 1

            # update HOF
            ind = self.environment.convert_action_to_gene(self.action)
            ind.fitness = Fitness((-sum(self.episode_rewards),))
            ind.ga_source = self.generated_by_ga
            self.hall_of_fame.update([ind])
            self.pop.popleft()
            self.pop.append(ind)

            # record source
            self.environment.source_list.append(self.generated_by_ga)

            self.state = self.next_state  # this is to set the state for the next iteration
            if self.config.verbose:
                print('state', self.state)
            self.episode_step_number += 1
        if self.time_to_learn():
            self.actor_learn()
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        if self.adaptive_hybrid:
            if self.adaptive_fun == None:
                raise Exception
            # automatically determine the ratio of GA based on historical successful rate
            counter = 0
            ga_list = []
            rl_list = []
            for x in self.pop:
                if not hasattr(x, 'ga_source'):
                    break
                if x.ga_source:
                    ga_list.append(x.fitness.wvalues[0])
                else:
                    rl_list.append(x.fitness.wvalues[0])
                counter += 1
            if counter == 90:
                self.ga_probability = self.adaptive_fun(self.ga_probability, self.initial_ga_probability,
                                                        np.array(ga_list), np.array(rl_list))
                # avoid deadlock (GA probability must greater than 0.05)
                self.ga_probability = max(self.ga_probability, 0.05)
            if self.config.verbose:
                print('GA probability', self.ga_probability)

        action, log_probabilities, all_log_distribution = self.pick_action_and_get_log_probabilities()
        if self.config.avoid_repetition:
            # resampling (avoid repetition)
            counter = 0
            while tuple(action) in self.history_action:
                if counter > 500:
                    raise Exception("deadlock!")
                action, log_probabilities, all_log_distribution = self.pick_action_and_get_log_probabilities()
                counter += 1

        self.history_action.add(tuple(action))
        self.store_log_probabilities(log_probabilities)
        if self.entropy_loss:
            self.all_log_distribution = all_log_distribution
        self.action = action
        self.conduct_action(action)

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using unsqueeze
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_probabilities = self.policy.forward(state).cpu()
        # action_probabilities = self.policy.forward(torch.tensor([[1.0]])).cpu()
        actions = []
        action_distributions = []
        all_log_distribution = []
        arity = self.environment.max_arity
        if random.random() >= self.ga_probability:
            # print('sampling from RL')
            self.generated_by_ga = False
            # sampling from RL
            try:
                remain_gene = 1
                for i in range(self.environment.gene_length):
                    if self.variable_length:
                        if remain_gene == 0:
                            break
                    single_action_len = self.action_size // self.environment.gene_length
                    prob = torch.ones(single_action_len)
                    if i < self.environment.head_length:
                        # not variable length
                        if not self.variable_length:
                            prob[len(self.environment.pset.functions):] = 0
                    else:
                        prob[:len(self.environment.pset.functions)] = 0
                    # TODO: RuntimeError: invalid multinomial distribution (encountering probability entry < 0)
                    softmax_result = torch.mul(
                        (torch.softmax(action_probabilities[0, single_action_len * i:single_action_len * (i + 1)],
                                       dim=0)),
                        prob)
                    # protected probability
                    # softmax_result = torch.max(softmax_result, torch.zeros_like(softmax_result))
                    # softmax_result = torch.nan_to_num_(softmax_result)
                    action_distribution = Categorical(softmax_result)

                    action = action_distribution.sample()

                    gene_point = action.item()
                    if gene_point < len(self.environment.pset.functions):
                        remain_gene += 2
                    remain_gene -= 1
                    actions.append(gene_point)
                    action_distributions.append(action_distribution.log_prob(action))

                    if self.entropy_loss:
                        if i < self.environment.head_length:
                            all_log_distribution.append(action_distribution.entropy())
                        else:
                            all_log_distribution.append(action_distribution.entropy())
            except:
                print('softmax_result', softmax_result)

        if self.chromosome_fusion:
            all_log_distribution.clear()

        if len(actions) == 0 or self.chromosome_fusion:
            # print('sampling from GA')
            self.generated_by_ga = True
            toolbox = self.toolbox

            if self.chromosome_fusion and len(actions) != 0:
                ind = self.environment.convert_action_to_gene(actions)
                ind.fitness = Fitness((-sum(self.episode_rewards),))
                offspring = [ind]

                # mutation
                for op in toolbox.pbs:
                    if op.startswith('mut'):
                        offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
            else:
                # sampling from GA
                offspring = selTournament(list(self.pop) + list(self.hall_of_fame), 2, tournsize=5)
                offspring = self.toolbox.clone(offspring)

                # mutation
                for op in toolbox.pbs:
                    if op.startswith('mut'):
                        offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

                # crossover
                for op in toolbox.pbs:
                    if op.startswith('cx'):
                        offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

            actions = self.environment.convert_gene_to_action(offspring[0])

            for i, action in enumerate(actions):
                single_action_len = self.action_size // self.environment.gene_length
                prob = torch.ones(single_action_len)
                if i < self.environment.head_length:
                    # not variable length
                    if not self.variable_length:
                        prob[len(self.environment.pset.functions):] = 0
                else:
                    prob[:len(self.environment.pset.functions)] = 0
                softmax_result = torch.mul(
                    (torch.softmax(action_probabilities[0, single_action_len * i:single_action_len * (i + 1)],
                                   dim=0)),
                    prob)
                action_distribution = Categorical(softmax_result)
                action_distributions.append(action_distribution.log_prob(torch.tensor(action)))

                if self.entropy_loss:
                    if i < self.environment.head_length:
                        all_log_distribution.append(action_distribution.entropy())
                    else:
                        all_log_distribution.append(action_distribution.entropy())

        return actions, action_distributions, all_log_distribution

    def store_log_probabilities(self, log_probabilities):
        """Stores the log probabilities of picked actions to be used for learning later"""
        for log in log_probabilities:
            self.episode_log_probabilities.append(log)

    def store_reward(self):
        """Stores the reward picked"""
        self.episode_rewards.append(self.reward)

    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        # baseline
        self.history_rewards.append(total_discounted_reward)
        # total_discounted_reward = total_discounted_reward - np.max(self.history_rewards)
        total_discounted_reward = total_discounted_reward - np.mean(self.history_rewards)
        if self.config.verbose:
            print('normalized', total_discounted_reward)

        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        if self.entropy_loss:
            if self.config.verbose:
                print('policy loss', policy_loss)
            for log in self.all_log_distribution:
                policy_loss -= self.regularization_ratio * torch.sum(log)
            if self.config.verbose:
                print('policy loss', policy_loss)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_episode_discounted_reward(self):
        """Calculates the cumulative discounted return for the episode"""
        discounts = self.hyperparameters["discount_rate"] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        """Calculates the loss from an episode"""
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            # gradient ascent, improve log probability
            # gradient descent, improve negative log probability
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.stack(policy_loss).sum()
        # We need to add up the losses across the mini-batch to get 1 overall loss
        return policy_loss

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done
