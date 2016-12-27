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
DISCOUNT = .90
N_EPISODES = 400
N_STEPS = 200
BATCH_SIZE = 100
LEARNING_RATE = 1.e-3
DROPOUT_KEEP_PROB = 1.0
RENDER = True

env = gym.make(env_name)
observation = env.reset()

n_state_descriptors = env.observation_space.shape[0]
n_possible_actions = env.action_space.n
HIDDEN_LAYER_SIZE = 200

""" We're going to represent our q-learning with a NN that takes n_state_descriptors as inputs and evaluates
n_possible actions as outputs. We pick the action with the best NN output score."""

# create our NN
model = Sequential()
input_layer = Dense(HIDDEN_LAYER_SIZE, input_dim=n_state_descriptors, activation="relu")
hidden_layer = Dense(n_possible_actions, activation="sigmoid")
model.add(input_layer)
model.add(Dropout(0.5))  # prevent overfitting
model.add(hidden_layer)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

sgdc = SGDClassifier()

actions = []
rewards = []
observations = []
policies = []
done = False
episode = 0


def to_values(rewards: Sequence) -> Sequence:
    discounted = list(rewards)
    for ii in range(1, len(reward) + 1):
        if ii == 1: discounted[-ii] = rewards[-ii]
        discounted[-ii] = rewards[-ii] + (DISCOUNT * discounted[1 - ii])
    return discounted


while not episode < N_EPISODES:
    if RENDER: env.render()

    policy = model.predict(np.matrix(observation))
    action = np.argmax(policy) # we want to upgrade this to sampling
    # logp = np.max(policy)
    observation, reward, done, info = env.step(action)

    observations.append(observation)
    actions.append(action)
    policies.append(policy)
    rewards.append(reward)

    if done:
        # if episode % BATCH_SIZE == 0:
        #     pass
        # update model
        values = to_values(rewards)
        values -= np.mean(values)
        values /= np.std(values)

        for ii in range(len(actions)):
            policies[ii][actions[ii]] *= values[ii]  # modulate the action we took by our advantage
        # lg.fit()
        sgdc.partial_fit(observations, values)

        episode += 1
        actions = []
        rewards = []
        observations = []
        policies = []
        observation = env.reset()
