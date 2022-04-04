import pygame
import random
import copy
import numpy as np
import itertools as it
import time

from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from Board import board
from Pacman import Pacman

from collections import deque


class DQNAgent:
    def __init__(self, action_size, learning_rate, model, get_legal_actions):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = learning_rate
        self.model = model
        self.get_legal_actions = get_legal_actions

    def remember(self, state, action, reward, next_state, done):
        # Function adds information to the memory about last action and its results
        state = state.transform_to_vector()
        next_state = next_state.transform_to_vector()
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        if self.get_legal_actions == None:
            possible_actions = [0, 1]
        else:
            possible_actions = self.get_legal_actions(state)
            # lst = [0,1]
        # choice = random.choices(lst, weights=(1-self.epsilon, self.epsilon))
        # chosen_action = random.choice(possible_actions) if choice == [1] else self.get_best_action(state)
        # return chosen_action

        return random.choice(possible_actions) if (np.random.random() <= self.epsilon) else self.get_best_action(state)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """
        state = state.transform_to_vector()
        state = np.expand_dims(state, axis=0)
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory.
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for chosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        """

        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states_batch = np.array([sample[0] for sample in minibatch])

        target_batch = self.model.predict(states_batch)

        next_states_batch = np.array([sample[3] for sample in minibatch])
        next_states_target_batch = self.model.predict(next_states_batch, batch_size=batch_size)

        for batch_idx, (state, action, reward, next_state, done) in enumerate(minibatch):

            if done:
                target_batch[batch_idx][action] = reward
            else:
                target_batch[batch_idx][action] = reward + self.gamma * max(next_states_target_batch[batch_idx][:])

        self.model.fit(states_batch, target_batch, batch_size=batch_size, verbose=0)

    def update_epsilon_value(self):
        # Every each epoch epsilon value should be updated according to equation:
        # self.epsilon *= self.epsilon_decay, but the updated value shouldn't be lower then epsilon_min value
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

clock = pygame.time.Clock()

pacman = Pacman(board)

pacman.reset()


def play(env, agent):
    env.turn_off_display()
    total_reward = 0
    state = env.reset()

    for time in range(1000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state
        if done:
            break

    return total_reward


def agent_test(env, agent, num_of_plays=10):
    env.turn_off_display()

    total_rewards = []
    for i in range(num_of_plays):
        total_rewards.append(play(env, agent))
    num_of_victories = len([r for r in total_rewards if r > 0])
    print("\nTotal victory ratio:", num_of_victories, "/", num_of_plays)
    return total_rewards


def play_and_display(env, agent):
    env.turn_on_display()
    total_reward = 0
    state = env.reset()

    for time in range(1000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        clock.tick(5)
        total_reward += reward

        state = next_state
        if done:
            break

    return total_reward


def train(env, agent, episodes=100):
    done = False
    batch_size = 64
    EPISODES = episodes
    counter = 0
    game_counter = 0
    MAX_MOVES_PER_GAME = 1000
    mean_rewards = []
    for e in range(EPISODES):
        start = time.time()
        summary = []
        for _ in range(100):
            total_reward = 0
            state = env.reset()

            for t in range(MAX_MOVES_PER_GAME):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # add to experience memory
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    game_counter += 1
                    break

            if len(agent.memory) > batch_size:
                agent.replay(64)

            summary.append(total_reward)
        agent.update_epsilon_value()
        end = time.time()
        mean_reward = np.mean(summary)
        mean_rewards.append(mean_reward)
        print("\nAchieved rewards:", summary)
        print(
            "epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}\ttime = {:.3f}\tgame num: {}\tbest mean reward = {:.3f}".format(
                e, mean_reward,
                agent.epsilon,
                end - start,
                game_counter,
                max(mean_rewards)))


'''
Deep Q-Learning
'''

state_size = pacman.get_state_size()
action_size = 4
learning_rate = 0.001

model = Sequential()
model.add(Dense(128, input_dim=state_size, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(action_size))
model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=learning_rate))

env = pacman

DISPLAY_MODE = False
weights_file = 'board_weights'

if DISPLAY_MODE:

    agent = DQNAgent(action_size, learning_rate, model,
                     get_legal_actions=pacman.get_possible_actions)
    pacman.turn_on_display()
    model.load_weights(weights_file)

    for i in range(10):
        play_and_display(pacman, agent)

else:
    pacman.turn_off_display()
    agent = DQNAgent(action_size, learning_rate, model,
                     get_legal_actions=pacman.get_possible_actions)
    # agent.set_new_epsilon_decay(0.9999)
    train(env, agent)
    model.save_weights(weights_file)
