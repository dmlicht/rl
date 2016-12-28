from collections import namedtuple
from typing import Sequence

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.linear_model import SGDClassifier

env_name = 'CartPole-v0'
MONITOR_FILE = 'results/' + env_name
LOG_DIR = 'tmp/keras/' + env_name
MAX_MEMORY = 40000
DISCOUNT = .98
N_EPISODES = 400
N_STEPS = 200
BATCH_SIZE = 2
# BATCH_SIZE = 3
LEARNING_RATE = 1.e-3
DROPOUT_KEEP_PROB = 1.0
RENDER = True

# HIDDEN_LAYER_SIZE = 200

""" We're going to represent our q-learning with a NN that takes n_state_descriptors as inputs and evaluates
n_possible actions as outputs. We pick the action with the best NN output score."""


# sgdc = SGDClassifier()


Episode = namedtuple("Episode", "observations policies actions rewards")


def main():
    env = gym.make(env_name)

    n_state_descriptors = env.observation_space.shape[0]
    n_possible_actions = env.action_space.n

    model = Sequential()
    model.add(Dense(128, input_dim=n_state_descriptors, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(128, input_dim=n_state_descriptors, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(n_possible_actions, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    episode = 0
    episodes = []

    while episode < N_EPISODES:
        observations, policies, actions, rewards = run_episode(env, model)
        episodes.append(Episode(observations, policies, actions, rewards))
        episode += 1
        if episode % BATCH_SIZE == 0:
            values = np.concatenate([to_values(ep.rewards) for ep in episodes])
            values -= np.mean(values)
            values /= np.std(values)

            batch_policies = np.concatenate([ep.policies for ep in episodes])
            batch_observations = np.concatenate([ep.observations for ep in episodes])
            batch_actions = np.concatenate([ep.actions for ep in episodes])

            for ii in range(len(batch_actions)):
                chosen_action = batch_actions[ii]
                batch_policies[ii, chosen_action] *= values[ii]  # modulate the action we took by our advantage
            model.train_on_batch(batch_observations, batch_policies)
            # lg.fit()
            # for ep in episodes:
            #     ep.values = to_values(ep.rewards)


        # policies = np.matrix(policies)  # convert to matrix
        # observations = np.matrix(observations)
        #
        # values = to_values(rewards)
        # values -= np.mean(values)
        # values /= np.std(values)
        #
        # for ii in range(len(actions)):
        #     chosen_action = actions[ii]
        #     policies[ii, chosen_action] *= values[ii]  # modulate the action we took by our advantage
        # # lg.fit()
        # model.train_on_batch(observations, policies)


def run_episode(env, model):
    done = False
    observation = env.reset()
    observations = []
    policies = []
    actions = []
    rewards = []
    turns = 0

    while not done:
        if RENDER: env.render()

        turns += 1

        policy = model.predict(np.matrix(observation))
        # action = np.argmax(policy)  # we want to upgrade this to sampling
        action = np.random.choice(range(len(policy[0])), p=policy[0])
        # logp = np.max(policy)
        observation, reward, done, info = env.step(action)

        observations.append(observation)
        actions.append(action)
        policies.append(policy[0])  # this return
        if done: # the reward is the length of playtime
            rewards.append(turns)
        else:
            rewards.append(0)
    return observations, policies, actions, rewards


def to_values(rewards: Sequence) -> Sequence:
    discounted = list(rewards)
    for ii in range(1, len(rewards) + 1):
        if ii == 1: discounted[-ii] = rewards[-ii]
        discounted[-ii] = rewards[-ii] + (DISCOUNT * discounted[1 - ii])
    return discounted


if __name__ == '__main__':
    main()
