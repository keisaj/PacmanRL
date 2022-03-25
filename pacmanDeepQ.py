import pygame
import random
import copy
import numpy as np
from collections import defaultdict
import itertools as it
import time

from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from collections import deque


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NO_MOVE = 4

class State:

    def __init__(self, player_pos:dict, ghost_pos:dict, food_pos:list, possible_actions:tuple = None):
        self.player_pos = player_pos
        self.ghost_pos = ghost_pos
        self.food_pos = food_pos
        self.possible_actions = possible_actions
        self.probability  = None
        self.name = None

    def transform_to_vector(self):
        vector = []
        vector.append([self.player_pos['x'], self.player_pos['y']])
        vector.append([self.ghost_pos['x'], self.ghost_pos['y']])
        for food in self.food_pos:
            vector.append([food['x'], food['y']])

        if len(self.food_pos) == 0:
            vector.append([-1, -1])
            vector.append([-1, -1])
        if len(self.food_pos) == 1:
            vector.append([-1, -1])

        return np.array(vector).flatten()

    def simulate_player_action(self, action:int) -> dict:
        pos = dict()
        if action == 0:
            pos['x'] = self.player_pos['x'] - 1
            pos['y'] = self.player_pos['y']
        if action == 1:
            pos['x'] = self.player_pos['x']
            pos['y'] = self.player_pos['y'] + 1
        if action == 2:
            pos['x'] = self.player_pos['x'] + 1
            pos['y'] = self.player_pos['y']
        if action == 3:
            pos['x'] = self.player_pos['x']
            pos['y'] = self.player_pos['y'] - 1
        if action == 4:
            pos['x'] = self.player_pos['x']
            pos['y'] = self.player_pos['y']
        return pos

    def simulate_ghost_action(self, action:int) -> dict:
        pos = dict()
        if action == 0:
            pos['x'] = self.ghost_pos['x'] - 1
            pos['y'] = self.ghost_pos['y']
        if action == 1:
            pos['x'] = self.ghost_pos['x']
            pos['y'] = self.ghost_pos['y'] + 1
        if action == 2:
            pos['x'] = self.ghost_pos['x'] + 1
            pos['y'] = self.ghost_pos['y']
        if action == 3:
            pos['x'] = self.ghost_pos['x']
            pos['y'] = self.ghost_pos['y'] - 1
        return pos


class Pacman:

    def __init__(self, board):
        """
            Pacman:
        """

        self.player_image = pygame.transform.scale(pygame.image.load("../../Desktop/RL/Pacman/assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("../../Desktop/RL/Pacman/assets/red_ghost.png"), (30, 30))

        self.display_mode_on = True

        self.board = board
        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
        self.score = 0
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == 'p':
                    self.player_pos['x'] = x
                    self.player_pos['y'] = y
                    self.init_player_pos = self.player_pos.copy()
                elif self.board[y][x] == 'g':
                    ghost = dict()
                    ghost['x'] = x
                    ghost['y'] = y
                    ghost['direction'] = random.choice([LEFT, DOWN])
                    self.ghosts.append(ghost)
                elif self.board[y][x] == '*':
                    food = dict()
                    food['x'] = x
                    food['y'] = y
                    self.foods.append(food)

        self.init_foods = copy.deepcopy(self.foods)
        self.init_ghosts = copy.deepcopy(self.ghosts)
        ######
        self.states = self.__init_states()
        print(len(self.states))
        ######
        self.__draw_board()
    def reset(self):
        """ resets state of the environment """
        self.foods = copy.deepcopy(self.init_foods)
        self.ghosts = copy.deepcopy(self.init_ghosts)
        self.player_pos = self.init_player_pos.copy()
        self.score = 0
        return self.__get_state()

    def __init_states(self) -> list:
        states = []
        possible_player_pos = []
        possible_ghost_pos = []
        # possible_food_pos = [[{'x':0, 'y':0}, {'x':2, 'y':2}], [{'x':0, 'y':0}], [{'x':2, 'y':2}], []]
        possible_food_pos = self.generate_possible_food_pos(self.foods)
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] != 'w':
                    possible_player_pos.append({'x': x, 'y': y})
                    possible_ghost_pos.append({'x': x, 'y': y})

        for player_pos in possible_player_pos:
            for ghost_pos in possible_ghost_pos:
                for food_pos in possible_food_pos:
                    states.append(State(player_pos=player_pos, ghost_pos=ghost_pos, food_pos=food_pos))

        for idx, state in enumerate(states):
            state.possible_actions = self.get_possible_actions(state)
            state.name = str(idx)
        return states

    def generate_possible_food_pos(self, foods):
        food_comb = []
        lst = []
        for i in range(len(foods) + 1):
            food_comb.append(it.combinations(foods, i))
        for elem in food_comb:
            for y in elem:
                lst.append(list(y))
        return lst

    def get_all_states(self):
        """ return a list of all possible states """
        return self.states

    def is_terminal(self, state) -> bool:
        """
        return true if state is terminal or false otherwise
        state is terminal when ghost is on the same position as pacman or all capsules are eaten
        """
        if (state.player_pos == state.ghost_pos) or state.food_pos == False:
            return True
        return False

    def get_possible_actions(self, state:object) -> tuple:
        """ return a tuple of possible actions in a given state """
        possible_actions = []
        player_x = state.player_pos['x']
        player_y = state.player_pos['y']
        directions = [[-1, 0], [0, 1], [1, 0], [0, -1]] # LEFT, DOWN, RIGHT, UP
        for idx, (x, y) in enumerate(directions):
            try:
                next_y = player_y + y
                next_x = player_x + x
                next_pos = self.board[next_y][next_x]
            except IndexError:
                continue
            else:
                if next_y != -1 and next_x != -1 and next_pos != 'w':
                    possible_actions.append(idx)

        return tuple(possible_actions)

    def get_next_states(self, state, action):
        """
        return a set of possible next states and probabilities of moving into them
        assume that ghost can move in each possible direction with the same probability, ghost cannot stay in place
        """
        # TODO nie biore pod uwage wiekszej ilosci duchow
        next_states = []
        for s in self.states:
            if s.player_pos == state.simulate_player_action(action) and s.food_pos == state.food_pos:
            # for ghost in s.ghost list:
                for i in range(4):
                    if s.ghost_pos == state.simulate_ghost_action(i):
                        next_states.append(s)
        try:
            probability = 1/len(next_states)
        except ZeroDivisionError:
            probability = 1.0
        return dict(zip(next_states,[probability]*len(next_states)))

    def get_reward(self, state, action, next_state):
        """
        return the reward after taking action in state and landing on next_state
            -1 for each step
            10 for eating capsule
            -500 for eating ghost
            500 for eating all capsules
        """
        reward = 0
        for food_pos in state.food_pos:
            if state.simulate_player_action(action) == food_pos:
                reward += 10
        if next_state.food_pos == False:
            reward += 500
        if state.simulate_player_action(action) == next_state.ghost_pos:
            reward -= 500
        reward -= 1
        return reward

    def step(self, action):
        '''
        Function apply action. Do not change this code
        :returns:
        state - current state of the game
        reward - reward received by taking action (-1 for each step, 10 for eating capsule, -500 for eating ghost, 500 for eating all capsules)
        done - True if it is end of the game, False otherwise
        score - temporarily score of the game, later it will be displayed on the screen
        '''

        width = len(self.board[0])
        height = len(self.board)

        # move player according to action

        if action == LEFT and self.player_pos['x'] > 0:
            if self.board[self.player_pos['y']][self.player_pos['x'] - 1] != 'w':
                self.player_pos['x'] -= 1
        if action == RIGHT and self.player_pos['x'] + 1 < width:
            if self.board[self.player_pos['y']][self.player_pos['x'] + 1] != 'w':
                self.player_pos['x'] += 1
        if action == UP and self.player_pos['y'] > 0:
            if self.board[self.player_pos['y'] - 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] -= 1
        if action == DOWN and self.player_pos['y'] + 1 < height:
            if self.board[self.player_pos['y'] + 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return  self.__get_state(), reward, True, self.score

        # check if player eats food

        for food in self.foods:
            if food['x'] == self.player_pos['x'] and food['y'] == self.player_pos['y']:
                self.score += 10
                reward = 10
                self.foods.remove(food)
                break
        else:
            self.score -= 1
            reward = -1

        # move ghosts
        for ghost in self.ghosts:
            moved = False
            ghost_moves = [LEFT, RIGHT, UP, DOWN]
            if ghost['x'] > 0 and self.board[ghost['y']][ghost['x'] - 1] != 'w':
                if ghost['direction'] == LEFT:
                    if RIGHT in ghost_moves:
                        ghost_moves.remove(RIGHT)
            else:
                if LEFT in ghost_moves:
                    ghost_moves.remove(LEFT)

            if ghost['x'] + 1 < width and self.board[ghost['y']][ghost['x'] + 1] != 'w':
                if ghost['direction'] == RIGHT:
                    if LEFT in ghost_moves:
                        ghost_moves.remove(LEFT)
            else:
                if RIGHT in ghost_moves:
                    ghost_moves.remove(RIGHT)

            if ghost['y'] > 0 and self.board[ghost['y'] - 1][ghost['x']] != 'w':
                if ghost['direction'] == UP:
                    if DOWN in ghost_moves:
                        ghost_moves.remove(DOWN)
            else:
                if UP in ghost_moves:
                    ghost_moves.remove(UP)

            if ghost['y'] + 1 < height and self.board[ghost['y'] + 1][ghost['x']] != 'w':
                if ghost['direction'] == DOWN:
                    if UP in ghost_moves:
                        ghost_moves.remove(UP)
            else:
                if DOWN in ghost_moves:
                    ghost_moves.remove(DOWN)

            ghost['direction'] = random.choice(ghost_moves)

            if ghost['direction'] == LEFT and ghost['x'] > 0:
                if self.board[ghost['y']][ghost['x'] - 1] != 'w':
                    ghost['x'] -= 1
            if ghost['direction'] == RIGHT and ghost['x'] + 1 < width:
                if self.board[ghost['y']][ghost['x'] + 1] != 'w':
                    ghost['x'] += 1
            if ghost['direction'] == UP and ghost['y'] > 0:
                if self.board[ghost['y'] - 1][ghost['x']] != 'w':
                    ghost['y'] -= 1
            if ghost['direction'] == DOWN and ghost['y'] + 1 < height:
                if self.board[ghost['y'] + 1][ghost['x']] != 'w':
                    ghost['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return  self.__get_state(), reward, True, self.score

        self.__draw_board()

        if len(self.foods) == 0:
            reward = 500
            self.score += 500

        return self.__get_state(), reward, len(self.foods) == 0, self.score

    def __draw_board(self):
        '''
        Function displays current state of the board. Do not change this code
        '''
        if self.display_mode_on:
            self.screen.fill((0, 0, 0))

            y = 0

            for line in board:
                x = 0
                for obj in line:
                    if obj == 'w':
                        color = (0, 255, 255)
                        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 60, 60))
                    x += 60
                y += 60

            color = (255, 255, 0)
            # pygame.draw.rect(self.screen, color, pygame.Rect(self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15, 30, 30))
            self.screen.blit(self.player_image, (self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15))

            color = (255, 0, 0)
            for ghost in self.ghosts:
                # pygame.draw.rect(self.screen, color, pygame.Rect(ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15, 30, 30))
                self.screen.blit(self.ghost_image,
                                 (ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15))

            color = (255, 255, 255)

            for food in self.foods:
                pygame.draw.ellipse(self.screen, color, pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10, 10))

            pygame.display.flip()

    def __get_state(self):
        '''
        Function returns current state of the game
        :return: state
        '''
        # TODO - nie brana jest pod uwage wieksza liczba duchów, kolejność jedzonek w liscie moze sprawiać ze if nie przejdzie...
        for state in self.states:
            if state.player_pos == self.player_pos and state.ghost_pos['x'] == self.ghosts[0]['x'] and state.ghost_pos['y'] == self.ghosts[0]['y'] and state.food_pos == self.foods:
                return state

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True

    # def get_state_size(self):
    #
    #     return self.__get_state().transform_to_vector().shape

    def get_state_size(self):
        return len(board) * len(board[0]) * 4


class DQNAgent:
    def __init__(self, action_size, learning_rate, model, get_legal_actions):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
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

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #
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

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #
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
        #
        # INSERT CODE HERE to train network
        #

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


board = ["*   g",
         " www ",
         " w*  ",
         " www ",
         "p    "]


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


def test(env, agent, num_of_plays=10):
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

            #
            # INSERT CODE HERE to prepare appropriate format of the state for network
            #


            for t in range(MAX_MOVES_PER_GAME):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                #
                # INSERT CODE HERE to prepare appropriate format of the next state for network
                #


                # add to experience memory
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    game_counter += 1
                    break

            #
            # INSERT CODE HERE to train network if in the memory is more samples then size of the batch
            #
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
model.add(Dense(action_size))  # wyjście
model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=learning_rate))

env = pacman


DISPLAY_MODE = True
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

