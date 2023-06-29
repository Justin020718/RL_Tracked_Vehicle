import copy
import random
import time

import numpy as np
import A_star_algorithm
import PPO_agent
import torch
import os

BATCH = 10
Actor_Path = 'model/Actor_Model.pth'
Critic_Path = 'model/Critic_Model.pth'
length_to_end = {}

def choose_action(pi):
    if pi[0] > pi[1] and pi[0] > pi[2] and pi[0] > pi[3]:
        return 'up'
    if pi[1] > pi[0] and pi[1] > pi[2] and pi[1] > pi[3]:
        return 'down'
    if pi[2] > pi[0] and pi[2] > pi[1] and pi[2] > pi[3]:
        return 'left'
    if pi[3] > pi[0] and pi[3] > pi[1] and pi[3] > pi[2]:
        return 'right'

def get_reward(memory, start, end):
    reward = 0
    pre_state = None
    start_to_end = length_to_end[tuple(start)]
    for state in memory:
        route_len = length_to_end[tuple(state)]
        if PPO_agent.heuristic(state, end) == 0:
            reward = reward + start_to_end
        if pre_state is not None:
            route_pre_len = length_to_end[tuple(pre_state)]
            if route_len < route_pre_len:
                reward = reward + 1.2-(route_len/start_to_end)**0.5
            elif route_len == route_pre_len:
                reward = reward - 0.2
            else:
                reward = reward - 0.01
        pre_state = state
    reward = reward - 0.1 * (len(memory) - len(set(memory)))
    return reward

def combined_train(grid):
    length = len(grid)
    width = len(grid[0])
    for i in range(BATCH):
        a = random.randint(0, length - 1)
        b = random.randint(0, width - 1)
        while grid[a][b] == 1:
            a = random.randint(0, length - 1)
            b = random.randint(0, width - 1)
        start = (a, b)
        a = random.randint(0, length - 1)
        b = random.randint(0, width - 1)
        while grid[a][b] == 1 and (a,b) == start:
            a = random.randint(0, length - 1)
            b = random.randint(0, width - 1)
        end = (a, b)
        if os.path.exists(Actor_Path) and os.path.exists(Critic_Path):
            PPO_agent.train(grid, start, end, torch.load(Actor_Path)['net'], torch.load(Critic_Path)['net'], 2*PPO_agent.heuristic(start, end))
        else:
            PPO_agent.train(grid, start, end, None, None, 2*PPO_agent.heuristic(start, end))

def solve(grid, start, end):
    #reward = 0
    flag = False
    step = 2048
    #start = 30, 30
    #combined_train(grid)
    #for i in range(BATCH):
    #    if os.path.exists(Actor_Path) and os.path.exists(Critic_Path):
    #        agent.train(grid, start, end, torch.load(Actor_Path)['net'], torch.load(Critic_Path)['net'], round(1000/(i+1)) + 2 * agent.heuristic(start, end))
    #    else:
    #        agent.train(grid, start, end, None, None, round(1000/(i+1)) + 2 * agent.heuristic(start, end))
    #PPO_agent.train(grid, start, end, torch.load(Actor_Path)['net'], torch.load(Critic_Path)['net'], 2**11)
    Actor = PPO_agent.actor()
    Actor.new_pi.load_state_dict(torch.load(Actor_Path)['net'])
    #memory = []
    #memory.append(tuple(start))

    state = PPO_agent.get_state(grid, start, end)
    pos = copy.deepcopy(start)
    node_path = []
    viewed_times = {}
    length = len(grid)
    width = len(grid[0])
    for i in range(length):
        for j in range(width):
            if grid[i][j] == 0:
                viewed_times[tuple((i, j))] = 0
    viewed_times[tuple(pos)] = 1
    hit_wall = 0
    for i in range(step):
        #if pos == end:
        #    return True, node_path
        node_path.append(tuple(pos))
        pi = Actor.new_pi.forward(torch.tensor(state, dtype=torch.float32))
        #action = PPO_agent.choose_action(pi)
        action = PPO_agent.choose_action(pi)
        #action = 'right' only a test, don't mind
        next = PPO_agent.do_action(grid, pos, action)
        while next == pos:
            hit_wall = hit_wall + 1
            action = PPO_agent.choose_action(pi)
            next = PPO_agent.do_action(grid, pos, action)
        if next == end:
            flag = True
            viewed_times[tuple(end)] = viewed_times[tuple(end)] + 1
            print('Found!step is {}'.format(i))
            break
        viewed_times[tuple(next)] = viewed_times[tuple(next)] + 1
        #memory.append(tuple(pos))
        #if len(memory) > PPO_agent.memory_size:
        #    memory.pop(0)
        #reward = reward + agent.get_reward(memory, start, end)
        state = PPO_agent.get_state(grid, next, end)
        pos = copy.deepcopy(next)
    #print("test_reward=",reward)
    #print(viewed_times)
    print('Totally hit wall {} times!'.format(hit_wall))
    print(flag)
    _sample_grid = np.ones(tuple([length,width]))
    if flag == True:
        for i in range(length):
            for j in range(width):
                if viewed_times.get(tuple([i,j])) is not None and viewed_times[tuple((i,j))] >= 1:
                    _sample_grid[i][j] = 0
        flag, shorter_node = A_star_algorithm.solve(_sample_grid, start, end)
    else:
        shortest = 10086
        shortest_node = tuple(start)
        for i in range(length):
            for j in range(width):
                if viewed_times.get(tuple([i,j])) is not None and viewed_times[tuple([i,j])] >= 1:
                    _sample_grid[i][j] = 0

        for i in range(length):
            for j in range(width):
                if viewed_times.get(tuple([i,j])) is not None and viewed_times[tuple([i,j])] >= 1:
                    distance = A_star_algorithm.heuristic([i,j], end)
                    if distance < shortest:
                        shortest = distance
                        shortest_node = [i,j]
        t1 = time.clock()
        _, shorter_node = A_star_algorithm.solve(_sample_grid, start, shortest_node)
        t2 = time.clock()
        t = t2 - t1
        state = PPO_agent.get_state(grid, shortest_node, end)
        pos = copy.deepcopy(shortest_node)
        node_path = []
        viewed_times = {}
        for i in range(length):
            for j in range(width):
                if grid[i][j] == 0:
                    viewed_times[tuple((i, j))] = 0
        viewed_times[tuple(pos)] = 1
        for i in range(step):
            pi = Actor.new_pi.forward(torch.tensor(state, dtype=torch.float32))
            #pi = [0.25, 0.25, 0.25, 0.25]
            action = PPO_agent.choose_action(pi)
            #action = 'right'
            node_path.append(tuple(pos))
            next = PPO_agent.do_action(grid, pos, action)
            if next == end:
                flag = True
                #print('Found!step is {}'.format(i))
                break
            viewed_times[tuple(next)] = viewed_times[tuple(next)] + 1
            state = PPO_agent.get_state(grid, next, end)
            pos = copy.deepcopy(next)
        shorter_node = shorter_node + node_path

    return True, shorter_node