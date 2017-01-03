from collections import namedtuple
from typing import Sequence

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import sgd
from keras.regularizers import WeightRegularizer
import sys
import random

env_name = 'CartPole-v0'
MONITOR_FILE = 'results/' + env_name
LOG_DIR = 'tmp/keras/' + env_name
N_EPISODES = 5000
BATCH_SIZE = 3
DISCOUNT = .98
EPISODE_BUFFER = 100
LEARNING_RATE = 1.e-3
EPSILON = 0.1
""" We're going to represent our q-learning with a NN that takes n_state_descriptors as inputs and evaluates
n_possible actions as outputs. We pick the action with the best NN output score."""

Episode = namedtuple("Episode", "observations policies actions rewards")


# def make_nn(ins: int, outs: int) -> Sequential:
#     model = Sequential()
    # model.add(Dense(64, input_dim=ins, activation="relu", W_regularizer=WeightRegularizer(l1=0.001, l2=0.001),
    #                 b_regularizer=WeightRegularizer(l1=0.5, l2=0.5)))
    # model.add(Dropout(.5))
    # model.add(Dense(32,  activation="relu", W_regularizer=WeightRegularizer(l1=0.001, l2=0.001),
    #                 b_regularizer=WeightRegularizer(l1=0.5, l2=0.5)))
    # model.add(Dropout(.5))
    # model.add(Dense(16, activation="relu"))
    # model.add(Dropout(.5))
    # model.add(Dense(outs, activation="softmax"))
    # model.add(Dense(outs))
    # model.add(Dense(outs, input_dim=ins, W_regularizer=WeightRegularizer(l1=0.001, l2=0.001),
    #                 b_regularizer=WeightRegularizer(l1=0.1, l2=0.1)))
    # model.compile(loss='mse', optimizer=sgd(lr=0.01), metrics=['accuracy'])
    # return model

def make_models(ins: int, outs: int):
    models = []
    for ii in range(outs):
        model = Sequential()
        model.add(Dense(8, input_dim=ins, activation="relu", b_regularizer=WeightRegularizer(l2=1)))
        model.add(Dropout(.5))
        model.add(Dense(1, W_regularizer=WeightRegularizer(l1=0.001, l2=0.001),
                        b_regularizer=WeightRegularizer(l2=1)))
        model.compile(loss='mse', optimizer=sgd(lr=0.0001), metrics=['accuracy'])
        models.append(model)
    return models

def softmax(ins: np.array) -> np.array:
    num = np.exp(ins)
    return num / np.sum(num)


class TDAgent:
    # def __init__(self, model):
    #     self.model = model

    def __init__(self, models):
        self.models = models

    def _predict(self, state):
        return [mm.predict(np.matrix(state))[0, 0] for mm in self.models]

    def _train(self, state, action, taken_action_value):
        self.models[action].train_on_batch(np.matrix(state), np.matrix(taken_action_value))
        # for ii, mm in enumerate(self.models):
        #     self.models[ii].train_on_batch(np.matrix(state), np.matrix(values[ii]))
            # self.model.train_on_batch(np.matrix(state), predicted_values)

    def _choose(self, qs):
        weights = softmax(qs)
        # return np.random.choice(range(len(qs)), p=weights)  # weighted sample
        if random.random() > EPSILON:
            return np.argmax(qs)  # we want to upgrade this to sampling
        else:
            return np.random.choice(len(qs))

    def act(self, state):
        # print(state)
        qs = self._predict(state)
        return self._choose(qs)


        # policy = self.model.predict(np.matrix(state))
        # weights = softmax(policy[0])
        # print(weights)
        # return np.random.choice(range(len(policy[0])), p=weights)  # weighted sample

    def reinforce(self, state, action, reward, new_state):
        # new_state_max_value = max(self.model.predict(np.matrix(new_state))[0])
        new_state_max_value = np.mean(self._predict(new_state))
        taken_action_value = reward + (DISCOUNT * new_state_max_value)
        # old_predictions = self.model.predict(np.matrix(state))
        # predicted_values = self.model.predict(np.matrix(state))
        # print(predicted_values[0], taken_action_value, action)
        old_predictions = self._predict(state)
        predicted_values = self._predict(state)
        predicted_values[action] = taken_action_value
        self._train(state, action, taken_action_value)
        # print(self.model.get_weights())
        # self.model.train_on_batch(np.matrix(state), predicted_values)
        # updated_predictions = self.model.predict(np.matrix(state))
        updated_predictions = self._predict(state)
        # print("before update: ", old_predictions, taken_action_value, action)
        # print("update values: ", predicted_values)
        # print("after update: ", updated_predictions, taken_action_value, action)
        # print("")

    def reinforce_done(self, state, action, reward):
        taken_action_value = -2
        # predicted_values = self.model.predict(np.matrix(state))
        predicted_values = self._predict(state)
        # print(predicted_values[0], taken_action_value, action)
        predicted_values[action] = taken_action_value
        # self._train(state, predicted_values)
        self._train(state, action, taken_action_value)
        # self.model.train_on_batch(np.matrix(state), predicted_values)


def main():
    env = gym.make(env_name)
    env.monitor.start(sys.argv[1], force=True)

    n_state_descriptors = env.observation_space.shape[0]
    n_possible_actions = env.action_space.n
    models = make_models(n_state_descriptors, n_possible_actions)
    agent = TDAgent(models)

    episodes = []

    for episode_ii in range(N_EPISODES):
        render = episode_ii % 20 == 0  # when to output how we're doing

        observations, policies, actions, rewards = run_episode(env, agent, render)

        episodes.append(Episode(observations, policies, actions, rewards))
        episodes = episodes[-EPISODE_BUFFER:]

        episode_ii += 1
        if render:
            print("episode: ", episode_ii, " length: ", np.mean([sum(ep.rewards) for ep in episodes]))

    env.monitor.close()


def run_episode(env, agent, render=False):
    done = False
    observation = env.reset()
    observations = []
    policies = []
    actions = []
    rewards = []

    while not done:
        if render: env.render()

        action = agent.act(observation)
        old_observation = observation
        # policy = model.predict(np.matrix(observation))
        # action = np.argmax(policy)  # we want to upgrade this to sampling
        # action = np.random.choice(range(len(policy[0])), p=policy[0])  # too sample
        observation, reward, done, info = env.step(action)
        if done:
            agent.reinforce_done(old_observation, action, reward)
        else:
            agent.reinforce(old_observation, action, reward, observation)

        observations.append(observation)
        actions.append(action)
        policies.append([])
        rewards.append(reward)
    return observations, policies, actions, rewards


def to_values(rewards: Sequence) -> Sequence:
    discounted = list(rewards)
    for ii in range(1, len(rewards) + 1):
        if ii == 1: discounted[-ii] = rewards[-ii]
        discounted[-ii] = rewards[-ii] + (DISCOUNT * discounted[1 - ii])
    return discounted


if __name__ == '__main__':
    main()
