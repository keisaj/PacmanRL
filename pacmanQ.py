import pygame
import random
from collections import defaultdict
from Board import board
from Pacman import Pacman


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    # ---------------------START OF YOUR CODE---------------------#

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
        return max(self.get_qvalue(state, action) for action in possible_actions)

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
        value = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (
                reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, value)

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
        lst = [0, 1]
        choice = random.choices(lst, weights=(1 - self.epsilon, self.epsilon))
        chosen_action = random.choice(possible_actions) if choice == [1] else self.get_best_action(state)
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


clock = pygame.time.Clock()

pacman = Pacman(board)

pacman.reset()

agent = QLearningAgent(alpha=0.5, epsilon=0.8, discount=0.99, get_legal_actions=pacman.get_possible_actions)

pacman.turn_off_display()

for i in range(10000):
    if i % 1000 == 0:
        print(f'Learning epoch: {i}')
    play_and_train(pacman, agent)

agent.turn_off_learning()
pacman.turn_on_display()

done = False

print("Game begins...")
state = pacman.reset()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    '''
    move pacman according to the policy from Value Iteration
    '''

    state, reward, done, score = pacman.step(agent.get_best_action(state))

    print(f'Score = {score}')
    clock.tick(1)
