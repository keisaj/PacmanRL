import pygame
import random
import copy
import numpy as np
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

        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))

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
        possible_food_pos = [[{'x':0, 'y':0}, {'x':2, 'y':2}], [{'x':0, 'y':0}], [{'x':2, 'y':2}], []]
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

        probability = 1/len(next_states)
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


board = ["*   g",
         " www ",
         " w*  ",
         " www ",
         "p    "]

clock = pygame.time.Clock()

pacman = Pacman(board)

pacman.reset()
'''
Apply Value Iteration algorithm for Pacman
'''

actions = [LEFT, DOWN, RIGHT, UP, NO_MOVE]
def value_iteration(mdp, gamma, theta):

    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state.name] = 0
        policy[current_state.name] = actions[0]

    #
    # INSERT CODE HERE to evaluate the best policy and value function for the given mdp
    #
    while True:
        delta = {k: 0 for k in V.keys()}
        for s in mdp.get_all_states():
            action_values = {}
            for a in mdp.get_possible_actions(s):
                val = 0
                for next_state, probability in mdp.get_next_states(s, a).items():
                    val += probability * (mdp.get_reward(s, a, next_state) + gamma * V[next_state.name])
                action_values[a] = val

            best_action = [k for k, v in action_values.items() if v == max(action_values.values())]

            policy[s.name] = np.random.choice(best_action)
            delta[s.name] = max(action_values.values()) - V[s.name]
            V[s.name] = max(action_values.values())
        print(f"Max delta = {max(delta.values())}")
        if max(delta.values()) < theta:
            print('Value Iteration algorithm is done...')
            break

    return policy, V

policy, V = value_iteration(pacman, 0.9, 0.01)

done = False
state, reward, done, score = pacman.step(RIGHT)
print("Game begins...")
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    '''
    move pacman according to the policy from Value Iteration
    '''

    state, reward, done, score = pacman.step(policy[state.name])

    print(f'Score = {score}')
    clock.tick(1)

