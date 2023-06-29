import os
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
import copy
import datetime
import random
from math import log
import A_star_algorithm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



logdir = "logs/dqn/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

memory_size = 3
EP_MAX = 1000
sample_MAX = 256
GAMA = 0.97
LR = 0.0001
E = 1.2
length_to_end = {}
BUFFER_CAPACITY = 2*17
BATCH_SIZE = 64
TARGET_UPDATE = 10

def calculate_length_to_end(grid, start, end):
    _, x = A_star_algorithm.solve(grid, start, end)
    if x is not None:
        length_to_end[tuple(start)] = len(x)
    else:
        length_to_end[tuple(start)] = 1000

class DuelingDQN(nn.Module):
    # in: state
    # out: Q = A + V
    def __init__(self, memory_size):
        super(DuelingDQN, self).__init__()
        self.layer1 = nn.Linear(8 * memory_size, 64)
        self.layer2 = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, 4)
        self.layer4 = nn.Linear(16, 1)
    def forward(self, state):
        state = state.to('cuda')
        state = self.layer1(state).to('cuda')
        state = torch.tanh(state).to('cuda')
        state = torch.tanh(self.layer2(state)).to('cuda')
        A = torch.tanh(self.layer3(state).to('cuda')).to('cuda')
        V = torch.tanh(self.layer4(state).to('cuda')).to('cuda')
        return A + V - torch.mean(A, dim=-1, keepdim=True).to('cuda')

def change_format(bs,grid,end):
    result = []
    for s in bs:
        result.append(get_state(grid,s,end))
    return result

def map_action(ba):
    result = []
    for a in ba:
        if a == 'up':
            result.append(0)
        elif a == 'down':
            result.append(1)
        elif a == 'left':
            result.append(2)
        else:
            result.append(3)
    return np.array(result).reshape(1,-1)

class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.buffer = []
        self.buffer_len = 0
        self.get_Q = DuelingDQN(memory_size).to('cuda')
        self.target = DuelingDQN(memory_size).to('cuda')
        #self.get_critic_V = critic_net().cuda()
        self.optimizer = torch.optim.Adam(self.get_Q.parameters(), lr=LR, eps=1e-5)
        self.lose_func = nn.MSELoss().to('cuda')
        self.learning_step = 0
        #self.lose_func = nn.MSELoss().cuda()

    def learn(self,grid,end):
        if self.learning_step % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.get_Q.state_dict())
        self.learning_step = self.learning_step + 1
        index = np.random.choice(BUFFER_CAPACITY, BATCH_SIZE)
        buffer = np.array(self.buffer)
        b_sample = buffer[index]
        b_s = torch.FloatTensor(change_format((b_sample[:,0]),grid,end)).to('cuda')
        b_a = torch.LongTensor(map_action(b_sample[:,1])).to('cuda')
        b_r = torch.FloatTensor(list(b_sample[:,2])).to('cuda')
        b_s_ = torch.FloatTensor(change_format((b_sample[:,3]),grid,end)).to('cuda')
        q_eval = self.get_Q(b_s).to('cuda')
        #print(q_eval)
        #print(b_a)
        q_eval = q_eval.gather(1,b_a).to('cuda')
        q_next = self.target(b_s_).detach().to('cuda')
        q_target = b_r + GAMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to('cuda')
        loss = self.lose_func(q_eval, q_target).to('cuda')
        writer.add_scalar('loss', loss, self.learning_step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def heuristic(start, end):
    (x, y) = start
    (x1, y1) = end
    return abs(x1 - x) + abs(y1 - y)

def explore_next_state(grid, node, end):
    next_state = []
    (x, y) = node
    x_max = np.size(grid, 0)
    y_max = np.size(grid, 1)
    next_state.append(0)
    next_state.append(0)
    next_state.append(0)
    next_state.append(0)
    if  grid[x+1][y] == 1 or x+1 < 0 or y < 0 or x+1 >= x_max - 1 or y >= y_max - 1: # right
        next_state[3] = 1
    if  grid[x-1][y] == 1 or x-1 < 0 or y < 0 or x-1 >= x_max - 1 or y >= y_max - 1: # left
        next_state[2] = 1
    if  grid[x][y+1] == 1 or x < 0 or y+1 < 0 or x >= x_max - 1 or y+1 >= y_max - 1: # up
        next_state[1] = 1
    if  grid[x][y-1] == 1 or x < 0 or y-1 < 0 or x >= x_max - 1 or y-1 >= y_max - 1: # down
        next_state[0] = 1
    return next_state

def choose_action(Q, epsilon):
    p = random.random()
    if p < epsilon:
        p = random.random()
        if p<=0.25:
            return 'up'
        elif p<=0.5:
            return 'down'
        elif p<=0.75:
            return 'left'
        else:
            return 'right'
    else:
        q_1, q_2, q_3, q_4 = Q
        if q_1 >= q_2 and q_1 >= q_3 and q_1 >= q_4:
            return 'up'
        if q_2 >= q_1 and q_2 >= q_3 and q_2 >= q_4:
            return 'down'
        if q_3 >= q_1 and q_3 >= q_2 and q_3 >= q_4:
            return 'left'
        if q_4 >= q_1 and q_4 >= q_2 and q_4 >= q_3:
            return 'right'

def do_action(grid, now, action):
    [x, y] = now
    if action == 'up':
        if grid[x][y + 1] == 0:
            return [x, y + 1]
        else:
            return now
    if action == 'down':
        if grid[x][y - 1] == 0:
            return [x, y - 1]
        else:
            return now
    if action == 'left':
        if grid[x - 1][y] == 0:
            return [x - 1, y]
        else:
            return now
    if action == 'right':
        if grid[x + 1][y] == 0:
            return [x + 1, y]
        else:
            return now

def get_state(grid, memory, end):
    state = []
    for pos in memory:
        state = state + explore_next_state(grid, pos, end)
        a, b = pos
        c, d = end
        state.append(a)
        state.append(b)
        state.append(c)
        state.append(d)
    while len(state) < 8 * memory_size:
        state = [0, 0, 0, 0, 0, 0, 0, 0] + state  # 在前面补0
    return state

def get_reward(memory, start, end):
    #return 0
    reward = 0
    pre_state = None
    start_to_end = length_to_end[tuple(start)]
    for state in memory:
        route_len = length_to_end[tuple(state)]
        if heuristic(state, end) == 0:
            reward = reward + start_to_end
        if pre_state is not None:
            route_pre_len = length_to_end[tuple(pre_state)]
            if route_len < route_pre_len:
                reward = reward + 1 - (route_len/start_to_end)**0.4
            elif route_len == route_pre_len:
                reward = reward - 0.2                         # 软惩罚撞墙
            else:
                reward = reward - 0.01
        pre_state = state
    reward = reward - 0.1 * (len(memory) - len(set(memory)))  # 惩罚绕圈
    return reward

def train(grid, start, end, model = None):
    global E
    length = len(grid)
    width = len(grid[0])
    for i in range(length):
        for j in range(width):
            if grid[i][j] == 0:
                calculate_length_to_end(grid, [i, j], end)
    print("Reward table has been calculated!")
    DDDQN = DQN()
    if model is not None:
        DDDQN.get_Q.load_state_dict(model)
        DDDQN.target.load_state_dict(model)
    for ep in range(EP_MAX):
        done = False
        now = start
        memory = []
        memory.append(tuple(start))
        this_ep_reward = 0
        for _ in range(sample_MAX):
            memory_now = copy.deepcopy(memory)
            action = choose_action(DDDQN.get_Q(torch.tensor(get_state(grid,memory,end),dtype=torch.float32)),E) # 已经包含epsilon-greedy
            E = E / 2 + 0.05
            next = do_action(grid, now, action)
            memory.append(tuple(next))
            now = next
            if len(memory) > memory_size:
                memory.pop(0)
            memory_next = copy.deepcopy(memory)
            reward = get_reward(memory, start, end)
            this_ep_reward = this_ep_reward + reward
            DDDQN.buffer.insert(DDDQN.buffer_len, [memory_now, action, reward, memory_next])        #存buffer
            DDDQN.buffer_len = (DDDQN.buffer_len + 1) % BUFFER_CAPACITY
            if len(DDDQN.buffer) >= BUFFER_CAPACITY:
                DDDQN.learn(grid, end)
            if next == end:
                done = True
                break
        writer.add_scalar('this_ep_reward', this_ep_reward, ep)
        if ep % 10 == 0:
            save = {'net': DDDQN.get_Q.state_dict(), 'i': ep}
            torch.save(save, "model\DQN_Model.pth")




