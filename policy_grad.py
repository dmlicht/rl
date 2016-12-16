import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

actions = []
rewards = []
observations = []
done = False
episode = 0
while not episode < N_EPISODES:
    if RENDER: env.render()

    policy = model.predict(np.matrix(observation))
    action = np.argmax(policy)
    observation, reward, done, info = env.step(action)

    actions.append(action)
    observations.append(observation)
    rewards.append(reward)

    if done:

        if episode % BATCH_SIZE == 0:
            pass
            # update model

        episode += 1
        observation = env.reset()
