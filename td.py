from collections import namedtuple
from typing import Sequence

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import sgd
import sys

env_name = 'CartPole-v0'
MONITOR_FILE = 'results/' + env_name
LOG_DIR = 'tmp/keras/' + env_name
N_EPISODES = 5000
BATCH_SIZE = 3
DISCOUNT = .98
EPISODE_BUFFER = 100
LEARNING_RATE = 1.e-3

""" We're going to represent our q-learning with a NN that takes n_state_descriptors as inputs and evaluates
n_possible actions as outputs. We pick the action with the best NN output score."""

Episode = namedtuple("Episode", "observations policies actions value")


def make_nn(ins: int, outs: int) -> Sequential:
    model = Sequential()
    model.add(Dense(16, input_dim=ins, activation="relu"))
    model.add(Dropout(.5))
    # model.add(Dense(16, activation="relu"))
    # model.add(Dropout(.5))
    # model.add(Dense(outs, activation="softmax"))
    model.add(Dense(outs))
    model.compile(loss='mse', optimizer=sgd(lr=0.01), metrics=['accuracy'])
    return model


def main():
    env = gym.make(env_name)
    env.monitor.start(sys.argv[1], force=True)

    n_state_descriptors = env.observation_space.shape[0]
    n_possible_actions = env.action_space.n
    model = make_nn(n_state_descriptors, n_possible_actions)

    episodes = []

    for episode_ii in range(N_EPISODES):
        render = episode_ii % 20 == 0  # when to output how we're doing

        observations, policies, actions, rewards = run_episode(env, model, render)

        episodes.append(Episode(observations, policies, actions, rewards))
        episodes = episodes[-EPISODE_BUFFER:]

        episode_ii += 1
        if render:
            print("episode: ", episode_ii, " length: ", rewards, "avg: ", np.mean([ep.value for ep in episodes]))

        if episode_ii > 2:

            # values = np.concatenate([to_values(ep.rewards) for ep in episodes])
            ep = episodes[-1]
            ep_len = len(ep.actions)
            values = [ep.value for ep in episodes]
            values -= np.mean(values)
            values /= np.std(values)

            ep = episodes[-1]
            ep_len = len(ep.actions)
            ep_relative_val = values[-1]
            # ep_relative_val_arr = np.ones(ep_len) * ep_relative_val
            policy_diff = np.matrix(ep.policies)
            for ii in range(len(ep.actions)):
                chosen_action = ep.actions[ii]
                change = .5
                if ep_relative_val < -.5:
                    change = 0
                elif ep_relative_val > .5:
                    change = 1
                policy_diff[ii, chosen_action] = change
            model.train_on_batch(np.matrix(ep.observations), policy_diff)

    env.monitor.close()


def run_episode(env, model, render=False):
    done = False
    observation = env.reset()
    observations = []
    policies = []
    actions = []
    rewards = []
    turns = 0

    while not done:
        if render: env.render()

        turns += 1

        policy = model.predict(np.matrix(observation))
        action = np.argmax(policy)  # we want to upgrade this to sampling
        # action = np.random.choice(range(len(policy[0])), p=policy[0])  # too sample
        observation, reward, done, info = env.step(action)

        observations.append(observation)
        actions.append(action)
        policies.append(policy[0])
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
