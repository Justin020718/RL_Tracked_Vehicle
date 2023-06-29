import random
import DDDQN_agent
import torch
import os

import PPO_agent

PATH = 'model/DQN_Model.pth'
length_to_end = {}

def get_reward(memory, start, end):
    reward = 0
    pre_state = None
    start_to_end = length_to_end[tuple(start)]
    for state in memory:
        route_len = length_to_end[tuple(state)]
        if DDDQN_agent.heuristic(state, end) == 0:
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

def solve(grid, start, end):
    #reward = 0
    step = 1024
    #start = 30, 30
    #combined_train(grid)
    #for i in range(BATCH):
    #    if os.path.exists(Actor_Path) and os.path.exists(Critic_Path):
    #        agent.train(grid, start, end, torch.load(Actor_Path)['net'], torch.load(Critic_Path)['net'], round(1000/(i+1)) + 2 * agent.heuristic(start, end))
    #    else:
    #        agent.train(grid, start, end, None, None, round(1000/(i+1)) + 2 * agent.heuristic(start, end))
    #DDDQN_agent.train(grid, start, end, torch.load(PATH)['net'])
    DDDQN = DDDQN_agent.DQN()
    DDDQN.get_Q.load_state_dict(torch.load(PATH)['net'])
    memory = []
    memory.append(tuple(start))
    state = DDDQN_agent.get_state(grid, memory, end)
    pos = start
    node_path = []
    node_path.append(tuple(pos))
    e=0.1
    for i in range(step):
        #if pos == end:
        #    return True, node_path
        Q = torch.softmax(DDDQN.get_Q(torch.tensor(state,dtype=torch.float32)), dim=-1)
        action = PPO_agent.choose_action(Q)
        next = DDDQN_agent.do_action(grid, pos, action)
        node_path.append(tuple(next))
        memory.append(tuple(pos))
        if len(memory) > DDDQN_agent.memory_size:
            memory.pop(0)
        if next == pos:
            e=0.2
        else:
            e=e/2
        pos = next
        state = DDDQN_agent.get_state(grid,memory,end)
    return True, node_path
