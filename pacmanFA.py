import pygame
import random
import copy
import numpy as np
from collections import defaultdict
import itertools as it

from Board import board
from Pacman import Pacman

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


# class State:
#
#     def __init__(self, player_pos: dict, ghost_pos: dict, food_pos: list, possible_actions: tuple = None):
#         self.player_pos = player_pos
#         self.ghost_pos = ghost_pos
#         self.food_pos = food_pos
#         self.possible_actions = possible_actions
#         self.probability = None
#         self.name = None
#
#     def simulate_player_action(self, action: int) -> dict:
#         pos = dict()
#         if action == 0:
#             pos['x'] = self.player_pos['x'] - 1
#             pos['y'] = self.player_pos['y']
#         if action == 1:
#             pos['x'] = self.player_pos['x']
#             pos['y'] = self.player_pos['y'] + 1
#         if action == 2:
#             pos['x'] = self.player_pos['x'] + 1
#             pos['y'] = self.player_pos['y']
#         if action == 3:
#             pos['x'] = self.player_pos['x']
#             pos['y'] = self.player_pos['y'] - 1
#         if action == 4:
#             pos['x'] = self.player_pos['x']
#             pos['y'] = self.player_pos['y']
#         return pos
#
#     def simulate_ghost_action(self, action: int) -> dict:
#         pos = dict()
#         if action == 0:
#             pos['x'] = self.ghost_pos['x'] - 1
#             pos['y'] = self.ghost_pos['y']
#         if action == 1:
#             pos['x'] = self.ghost_pos['x']
#             pos['y'] = self.ghost_pos['y'] + 1
#         if action == 2:
#             pos['x'] = self.ghost_pos['x'] + 1
#             pos['y'] = self.ghost_pos['y']
#         if action == 3:
#             pos['x'] = self.ghost_pos['x']
#             pos['y'] = self.ghost_pos['y'] - 1
#         return pos
#
#
# class Pacman:
#
#     def __init__(self, board):
#         """
#             Pacman:
#         """
#
#         self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
#         self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))
#
#         self.display_mode_on = True
#
#         self.board = board
#         self.cell_size = 60
#         pygame.init()
#         self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
#         self.player_pos = dict()
#         self.ghosts = []
#         self.foods = []
#         self.score = 0
#         for y in range(len(self.board)):
#             for x in range(len(self.board[0])):
#                 if self.board[y][x] == 'p':
#                     self.player_pos['x'] = x
#                     self.player_pos['y'] = y
#                     self.init_player_pos = self.player_pos.copy()
#                 elif self.board[y][x] == 'g':
#                     ghost = dict()
#                     ghost['x'] = x
#                     ghost['y'] = y
#                     ghost['direction'] = random.choice([LEFT, DOWN])
#                     self.ghosts.append(ghost)
#                 elif self.board[y][x] == '*':
#                     food = dict()
#                     food['x'] = x
#                     food['y'] = y
#                     self.foods.append(food)
#
#         self.init_foods = copy.deepcopy(self.foods)
#         self.init_ghosts = copy.deepcopy(self.ghosts)
#         ######
#         self.states = self.__init_states()
#         print(len(self.states))
#         ######
#         self.__draw_board()
#         self.current_state = self.__get_state()
#
#     def reset(self):
#         """ resets state of the environment """
#         self.foods = copy.deepcopy(self.init_foods)
#         self.ghosts = copy.deepcopy(self.init_ghosts)
#         self.player_pos = self.init_player_pos.copy()
#         self.score = 0
#         return self.__get_state()
#
#     def __init_states(self) -> list:
#         states = []
#         possible_player_pos = []
#         possible_ghost_pos = []
#         # possible_food_pos = [[{'x':0, 'y':0}, {'x':2, 'y':2}], [{'x':0, 'y':0}], [{'x':2, 'y':2}], []]
#         possible_food_pos = self.generate_possible_food_pos(self.foods)
#         for y in range(len(self.board)):
#             for x in range(len(self.board[0])):
#                 if self.board[y][x] != 'w':
#                     possible_player_pos.append({'x': x, 'y': y})
#                     possible_ghost_pos.append({'x': x, 'y': y})
#
#         for player_pos in possible_player_pos:
#             for ghost_pos in possible_ghost_pos:
#                 for food_pos in possible_food_pos:
#                     states.append(State(player_pos=player_pos, ghost_pos=ghost_pos, food_pos=food_pos))
#
#         for idx, state in enumerate(states):
#             state.possible_actions = self.get_possible_actions(state)
#             state.name = str(idx)
#         return states
#
#     def generate_possible_food_pos(self, foods):
#         food_comb = []
#         lst = []
#         for i in range(len(foods) + 1):
#             food_comb.append(it.combinations(foods, i))
#         for elem in food_comb:
#             for y in elem:
#                 lst.append(list(y))
#         return lst
#
#     def get_all_states(self):
#         """ return a list of all possible states """
#         return self.states
#
#     def is_terminal(self, state) -> bool:
#         """
#         return true if state is terminal or false otherwise
#         state is terminal when ghost is on the same position as pacman or all capsules are eaten
#         """
#         if (state.player_pos == state.ghost_pos) or state.food_pos == False:
#             return True
#         return False
#
#     def get_possible_actions(self, state: object) -> tuple:
#         """ return a tuple of possible actions in a given state """
#         possible_actions = []
#         player_x = state.player_pos['x']
#         player_y = state.player_pos['y']
#         directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # LEFT, DOWN, RIGHT, UP
#         for idx, (x, y) in enumerate(directions):
#             try:
#                 next_y = player_y + y
#                 next_x = player_x + x
#                 next_pos = self.board[next_y][next_x]
#             except IndexError:
#                 continue
#             else:
#                 if next_y != -1 and next_x != -1 and next_pos != 'w':
#                     possible_actions.append(idx)
#
#         return tuple(possible_actions)
#
#     def get_next_states(self, state, action):
#         """
#         return a set of possible next states and probabilities of moving into them
#         assume that ghost can move in each possible direction with the same probability, ghost cannot stay in place
#         """
#         next_states = []
#         for s in self.states:
#             if s.player_pos == state.simulate_player_action(action) and s.food_pos == state.food_pos:
#                 # for ghost in s.ghost list:
#                 for i in range(4):
#                     if s.ghost_pos == state.simulate_ghost_action(i):
#                         next_states.append(s)
#
#         try:
#             probability = 1 / len(next_states)
#         except ZeroDivisionError:
#             probability = 0.0
#         return dict(zip(next_states, [probability] * len(next_states)))
#
#     def get_reward(self, state, action, next_state):
#         """
#         return the reward after taking action in state and landing on next_state
#             -1 for each step
#             10 for eating capsule
#             -500 for eating ghost
#             500 for eating all capsules
#         """
#         reward = 0
#         for food_pos in state.food_pos:
#             if state.simulate_player_action(action) == food_pos:
#                 reward += 10
#         if next_state.food_pos == False:
#             reward += 500
#         if state.simulate_player_action(action) == next_state.ghost_pos:
#             reward -= 500
#         reward -= 1
#         return reward
#
#     def step(self, action):
#         '''
#         Function apply action. Do not change this code
#         :returns:
#         state - current state of the game
#         reward - reward received by taking action (-1 for each step, 10 for eating capsule, -500 for eating ghost, 500 for eating all capsules)
#         done - True if it is end of the game, False otherwise
#         score - temporarily score of the game, later it will be displayed on the screen
#         '''
#
#         width = len(self.board[0])
#         height = len(self.board)
#
#         # move player according to action
#
#         if action == LEFT and self.player_pos['x'] > 0:
#             if self.board[self.player_pos['y']][self.player_pos['x'] - 1] != 'w':
#                 self.player_pos['x'] -= 1
#         if action == RIGHT and self.player_pos['x'] + 1 < width:
#             if self.board[self.player_pos['y']][self.player_pos['x'] + 1] != 'w':
#                 self.player_pos['x'] += 1
#         if action == UP and self.player_pos['y'] > 0:
#             if self.board[self.player_pos['y'] - 1][self.player_pos['x']] != 'w':
#                 self.player_pos['y'] -= 1
#         if action == DOWN and self.player_pos['y'] + 1 < height:
#             if self.board[self.player_pos['y'] + 1][self.player_pos['x']] != 'w':
#                 self.player_pos['y'] += 1
#
#         for ghost in self.ghosts:
#             if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
#                 self.score -= 500
#                 reward = -500
#                 self.__draw_board()
#                 return self.__get_state(), reward, True, self.score
#
#         # check if player eats food
#
#         for food in self.foods:
#             if food['x'] == self.player_pos['x'] and food['y'] == self.player_pos['y']:
#                 self.score += 10
#                 reward = 10
#                 self.foods.remove(food)
#                 break
#         else:
#             self.score -= 1
#             reward = -1
#
#         # move ghosts
#         for ghost in self.ghosts:
#             moved = False
#             ghost_moves = [LEFT, RIGHT, UP, DOWN]
#             if ghost['x'] > 0 and self.board[ghost['y']][ghost['x'] - 1] != 'w':
#                 if ghost['direction'] == LEFT:
#                     if RIGHT in ghost_moves:
#                         ghost_moves.remove(RIGHT)
#             else:
#                 if LEFT in ghost_moves:
#                     ghost_moves.remove(LEFT)
#
#             if ghost['x'] + 1 < width and self.board[ghost['y']][ghost['x'] + 1] != 'w':
#                 if ghost['direction'] == RIGHT:
#                     if LEFT in ghost_moves:
#                         ghost_moves.remove(LEFT)
#             else:
#                 if RIGHT in ghost_moves:
#                     ghost_moves.remove(RIGHT)
#
#             if ghost['y'] > 0 and self.board[ghost['y'] - 1][ghost['x']] != 'w':
#                 if ghost['direction'] == UP:
#                     if DOWN in ghost_moves:
#                         ghost_moves.remove(DOWN)
#             else:
#                 if UP in ghost_moves:
#                     ghost_moves.remove(UP)
#
#             if ghost['y'] + 1 < height and self.board[ghost['y'] + 1][ghost['x']] != 'w':
#                 if ghost['direction'] == DOWN:
#                     if UP in ghost_moves:
#                         ghost_moves.remove(UP)
#             else:
#                 if DOWN in ghost_moves:
#                     ghost_moves.remove(DOWN)
#
#             ghost['direction'] = random.choice(ghost_moves)
#
#             if ghost['direction'] == LEFT and ghost['x'] > 0:
#                 if self.board[ghost['y']][ghost['x'] - 1] != 'w':
#                     ghost['x'] -= 1
#             if ghost['direction'] == RIGHT and ghost['x'] + 1 < width:
#                 if self.board[ghost['y']][ghost['x'] + 1] != 'w':
#                     ghost['x'] += 1
#             if ghost['direction'] == UP and ghost['y'] > 0:
#                 if self.board[ghost['y'] - 1][ghost['x']] != 'w':
#                     ghost['y'] -= 1
#             if ghost['direction'] == DOWN and ghost['y'] + 1 < height:
#                 if self.board[ghost['y'] + 1][ghost['x']] != 'w':
#                     ghost['y'] += 1
#
#         for ghost in self.ghosts:
#             if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
#                 self.score -= 500
#                 reward = -500
#                 self.__draw_board()
#                 return self.__get_state(), reward, True, self.score
#
#         self.__draw_board()
#
#         if len(self.foods) == 0:
#             reward = 500
#             self.score += 500
#
#         return self.__get_state(), reward, len(self.foods) == 0, self.score
#
#     def __draw_board(self):
#         '''
#         Function displays current state of the board. Do not change this code
#         '''
#         if self.display_mode_on:
#             self.screen.fill((0, 0, 0))
#
#             y = 0
#
#             for line in board:
#                 x = 0
#                 for obj in line:
#                     if obj == 'w':
#                         color = (0, 255, 255)
#                         pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 60, 60))
#                     x += 60
#                 y += 60
#
#             color = (255, 255, 0)
#             # pygame.draw.rect(self.screen, color, pygame.Rect(self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15, 30, 30))
#             self.screen.blit(self.player_image,
#                              (self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15))
#
#             color = (255, 0, 0)
#             for ghost in self.ghosts:
#                 # pygame.draw.rect(self.screen, color, pygame.Rect(ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15, 30, 30))
#                 self.screen.blit(self.ghost_image,
#                                  (ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15))
#
#             color = (255, 255, 255)
#
#             for food in self.foods:
#                 pygame.draw.ellipse(self.screen, color,
#                                     pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10,
#                                                 10))
#
#             pygame.display.flip()
#
#     def __get_state(self):
#         '''
#         Function returns current state of the game
#         :return: state
#         '''
#         for state in self.states:
#             if state.player_pos == self.player_pos and state.ghost_pos['x'] == self.ghosts[0]['x'] and state.ghost_pos[
#                 'y'] == self.ghosts[0]['y'] and state.food_pos == self.foods:
#                 return state
#
#     def turn_off_display(self):
#         self.display_mode_on = False
#
#     def turn_on_display(self):
#         self.display_mode_on = True


class FunctionAproximationAgent:
    def __init__(self, alpha, epsilon, discount, env):

        self.env = env
        self.get_legal_actions = env.get_possible_actions
        self.weights = np.array([0.0, 0.0])

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    # ---------------------START OF YOUR CODE---------------------#

    def get_near_ghost_distance(self, player_pos, ghost_position):
        return self.bfs(self.env.board, player_pos, ghost_position)

    def get_near_food_distance(self, player_position, foods_position):
        results = []
        for food_position in foods_position:
            results.append(self.bfs(self.env.board, player_position, food_position))
        if not results:
            return 0
        return min(results)

    def bfs(self, graph, node, searched_node):
        if node == searched_node:
            return 0
        visited = [node]
        queue = [node, '|']
        steps = 1

        while queue:
            s = queue.pop(0)

            if s == '|':
                steps += 1
                queue.append('|')
                continue

            neighbours = []
            if -1 < s['x'] + 1 < 5 and graph[s['y']][s['x'] + 1] != 'w':
                neighbours.append({'x': s['x'] + 1, 'y': s['y']})
            if -1 < s['x'] - 1 < 5 and graph[s['y']][s['x'] - 1] != 'w':
                neighbours.append({'x': s['x'] - 1, 'y': s['y']})
            if -1 < s['y'] + 1 < 5 and graph[s['y'] + 1][s['x']] != 'w':
                neighbours.append({'x': s['x'], 'y': s['y'] + 1})
            if -1 < s['y'] - 1 < 5 and graph[s['y'] - 1][s['x']] != 'w':
                neighbours.append({'x': s['x'], 'y': s['y'] - 1})

            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)
                    if neighbour == searched_node:
                        return steps

    def get_features(self, state, action):
        possible_states = self.env.get_next_states(state, action)
        result = np.zeros(2)
        if possible_states:
            near_food_distance = self.get_near_food_distance(list(possible_states.keys())[0].player_pos, state.food_pos)
            near_ghost_distance = min(
                [self.get_near_ghost_distance(state.player_pos, state.ghost_pos) for state in possible_states])
            result += np.array([near_food_distance, near_ghost_distance])
        else:
            near_food_distance = self.get_near_food_distance(state.player_pos, state.food_pos)
            near_ghost_distance = self.get_near_ghost_distance(state.player_pos, state.ghost_pos)
            result += np.array([near_food_distance, near_ghost_distance])
        return result / 11

    def get_qvalue(self, state, action):
        feature_vector = self.get_features(state, action)
        return feature_vector @ self.weights.T

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #
        q = {action: self.get_qvalue(state, action) for action in possible_actions}
        max_value = max(q.values())

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #

        delta = (reward + gamma * self.get_value(next_state)) - self.get_qvalue(state, action)
        self.weights += learning_rate * delta * self.get_features(state, action)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #
        best_value = self.get_value(state)
        best_actions = [action for action in possible_actions if self.get_qvalue(state, action) == best_value]
        best_action = random.choice(best_actions)

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #
        if random.uniform(0, 1) <= epsilon:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        #
        # INSERT CODE HERE to train (update) agent for state
        #
        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward


# board = ["*   g",
#          " www ",
#          " w*  ",
#          " www ",
#          "p    "]

clock = pygame.time.Clock()

pacman = Pacman(board)

pacman.reset()
'''
Apply algorithm for Pacman
'''
alpha = 0.5
epsilon = 0.9
discount = 0.9

agent = FunctionAproximationAgent(alpha=alpha, epsilon=epsilon, discount=discount, env=pacman)

pacman.turn_off_display()
epochs = 1000
for i in range(epochs):
    if i % 100 == 0:
        print(f"Epoch: {i}/{epochs}")
    play_and_train(pacman, agent)

agent.turn_off_learning()
summary = []
for i in range(100):
    done = False

    pacman.reset()
    pacman.turn_on_display()
    state = pacman.current_state
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        state, reward, done, score = pacman.step(agent.get_best_action(state))
        print(f"Features: {agent.get_features(state, agent.get_best_action(state))}")
        # print(f"Score: {score}")
        summary.append(score)
        clock.tick(1)
print(f"Average score: {sum(summary) / len(summary)}")
