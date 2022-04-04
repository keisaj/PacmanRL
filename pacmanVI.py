import pygame
import numpy as np
from Board import board
from Pacman import Pacman, actions

clock = pygame.time.Clock()

pacman = Pacman(board)

pacman.reset()

def value_iteration(mdp, gamma, theta):

    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state.name] = 0
        policy[current_state.name] = actions[0]

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
state, reward, done, score = pacman.step(actions[2])
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

