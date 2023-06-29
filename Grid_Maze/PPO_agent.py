import copy
import datetime
import random
import math

from torch.distributions import Categorical

import A_star_algorithm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '1'
logdir = "logs/ppo/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

memory_size = 1
EP_MAX = 100000
#SAMPLE_NUM = 200
GAMA = 0.996
Actor_LR = 0.0001
Critic_LR = 0.0003
Actor_Update_Times = 10
Critic_Update_Times = 10
length_to_end = {}

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
    return np.array(result).reshape(-1,1)

def calculate_length_to_end(grid, start, end):
    _, x = A_star_algorithm.solve(grid, start, end)
    if x is not None:
        length_to_end[tuple(start)] = len(x)
    else:
        length_to_end[tuple(start)] = 1000

class actor_net(nn.Module):
    # in: 1-D vector[1,1,1,1,x,y,x1,y1]x4 up down left right(whether is wall, 1 means yes) and 1-D vector[x,y,x1,y1], positions of now and end
    # agent tries to memorize 4 previous states
    # out: 1-D vector[a,b,c,d] probabilities of each action
    def __init__(self,memory_size):
        super(actor_net, self).__init__()
        self.layer1 = nn.Linear(8 * memory_size, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 4)
        self.layer4 = nn.Softmax(0)
    def forward(self, state):
        #state = state
        state = torch.flatten(state)
        state = torch.tanh(self.layer1(state))
        state = torch.tanh(self.layer2(state))
        state = torch.tanh(self.layer3(state))
        return self.layer4(state)

class actor():
    def __init__(self):
        super(actor, self).__init__()
        self.old_pi = actor_net(memory_size)
        #self.old_pi = actor_net().cuda()
        self.new_pi = actor_net(memory_size)
        #self.new_pi = actor_net().cuda()
        self.entropy_coef = 0
        self.optimizer = torch.optim.Adam(self.new_pi.parameters(), lr=Actor_LR, eps=1e-5)

    def entropy(self, pis):
        entropy = 0
        entropy_tensor = []
        for pi in pis:
            for item in pi:
                entropy = entropy - item * np.log(item)
            entropy_tensor.append(entropy)
        entropy_tensor = torch.FloatTensor(entropy_tensor)
        return entropy_tensor

    def learn(self, td_e, pi_old, pi_new, actions, states): # td_e is the Advantage function
        for _ in range(Actor_Update_Times):
            ratio = []
            for i,action in enumerate(actions):
                this_pi_old = pi_old[i]
                this_pi_new = pi_new[i]
                if action == 'up':
                    ratio.append(this_pi_new[0].detach() / this_pi_old[0].detach())
                elif action == 'down':
                    ratio.append(this_pi_new[1].detach() / this_pi_old[1].detach())
                elif action == 'left':
                    ratio.append(this_pi_new[2].detach() / this_pi_old[2].detach())
                else:
                    ratio.append(this_pi_new[3].detach() / this_pi_old[3].detach())
            ratio=torch.FloatTensor(ratio)
            loss1 = (td_e * ratio)
            loss2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * td_e
            actor_loss = -torch.min(loss1, loss2)
            actor_loss = torch.sum(actor_loss - self.entropy_coef * Categorical(torch.stack(pi_new)).entropy())
            self.optimizer.zero_grad()
            actor_loss.requires_grad_(True)
            actor_loss.backward()
            self.optimizer.step()

            pi_new = []
            for state in states:
                pi_new.append(self.new_pi.forward(torch.tensor(state, dtype=torch.float32)))


    def update_old(self):
        self.old_pi.load_state_dict(self.new_pi.state_dict())

class critic_net(nn.Module):
    # in: state
    # out: V
    def __init__(self,memory_size):
        super(critic_net, self).__init__()
        self.layer1 = nn.Linear(8 * memory_size, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 8)
        self.layer4 = nn.Linear(8, 1)
    def forward(self, state):
        #state=state
        state = torch.tanh(self.layer1(state))
        state = torch.tanh(self.layer2(state))
        state = torch.tanh(self.layer3(state))
        return torch.tanh(self.layer4(state))

class critic():

    def __init__(self):
        super(critic, self).__init__()
        self.get_critic_V = critic_net(memory_size)
        #self.get_critic_V = critic_net().cuda()
        self.optimizer = torch.optim.Adam(self.get_critic_V.parameters(), lr=Critic_LR, eps=1e-5)
        self.lose_func = nn.MSELoss()
        #self.lose_func = nn.MSELoss().cuda()

    def learn(self, samples, ep):
        v_Vt = []
        v_Vt1 = []
        rewards = []
        for sample in samples:
            rewards.append(sample[3])
        rewards = torch.tensor([t for t in rewards], dtype = torch.float).view(-1,1)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        sample = samples[-1]

        V_t1_predict = self.get_critic_V(torch.tensor(sample[5], dtype=torch.float32))
        for i, sample in reversed(list(enumerate(samples))):
            v_Vt.append(V_t1_predict)
            V_t1_predict = rewards[i] + GAMA * V_t1_predict
        v_Vt.reverse()
        v_Vt = torch.tensor(v_Vt, dtype=torch.float32)

        for _ in range(Critic_Update_Times):
            self.optimizer.zero_grad()
            for sample in samples:
                v_Vt1.append(self.get_critic_V(torch.tensor(sample[1], dtype=torch.float32)))
            v_Vt1 = torch.tensor(v_Vt1, dtype=torch.float32)

            td_e = self.lose_func(v_Vt, v_Vt1)
            writer.add_scalar("Critic loss", td_e,ep)
            #td_e = (td_e - td_e.mean()) / (td_e.std() + 1e-7)
            td_e.requires_grad_(True)
            td_e.backward()
            self.optimizer.step()
            if _ != Critic_Update_Times - 1:
                v_Vt1 = []
        return (v_Vt-v_Vt1).detach()

def heuristic(start, end):
    (x, y) = start
    (x1, y1) = end
    return abs(x1 - x) + abs(y1 - y)

def explore_next_state(grid, node, end):
    if node == end:
        return None
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

def choose_action(pi):
    p = random.random()
    if p<=pi[0]:
        return 'up'
    elif p<=pi[0]+pi[1]:
        return 'down'
    elif p<=pi[0]+pi[1]+pi[2]:
        return 'left'
    else:
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

def get_state(grid, now, end):
    if now == end:
        c,d = end
        return [0,0,0,0,c,d,c,d]
    state=explore_next_state(grid,now,end)
    a,b=now
    c,d=end
    state.append(a)
    state.append(b)
    state.append(c)
    state.append(d)
    #for start in memory:
    #    state = state + (explore_next_state(grid, start, end))
    #    a, b = start
    #    c, d = end
    #    state.append(a)
    #    state.append(b)
    #    state.append(c)
    #    state.append(d)
    #while len(state) < 8 * memory_size:
    #    state = [0, 0, 0, 0, 0, 0, 0, 0] + state  # 在前面补0
    return state

def get_reward(now, next, start, end, viewed_times, t):
    #return 0
    start_to_end = length_to_end[tuple(start)]
    if heuristic(next,end) == 0:
        return 4 * start_to_end
    reward = 0
    route_pre_len = length_to_end[tuple(now)]
    route_len = length_to_end[tuple(next)]
    if now == next:
        reward = reward - 0.25
    elif route_pre_len > route_len:
        reward = reward + 1 - (route_len/start_to_end)**0.4
    else:
        reward = reward - 0.05
    reward = reward - np.clip(0.2 * (viewed_times[tuple(next)] - 2), -0.2 * math.exp(-t/256), 0.8)
    #reward = reward - 0.05
    return reward
    #pre_state = None

    # for state in memory:
    #     route_len = length_to_end[tuple(state)]
    #     if heuristic(state, end) == 0:
    #         reward = reward + 10 * start_to_end
    #     if pre_state is not None:
    #         route_pre_len = length_to_end[tuple(pre_state)]
    #         if route_len < route_pre_len:
    #             reward = reward + 2 - (route_len/start_to_end)**0.4
    #         elif route_len == route_pre_len:
    #             reward = reward - 0.2                         # 软惩罚撞墙
    #         else:
    #             reward = reward - 0.05
    #     pre_state = state
    #reward = reward - 0.2 * (len(memory) - len(set(memory)))  # 惩罚绕圈
    #return reward


def train(grid, start, end, actor_model = None, critic_model = None, SAMPLE_NUM = 200):
    length = len(grid)
    width = len(grid[0])
    for i in range(length):
        for j in range(width):
            if grid[i][j] == 0:
                calculate_length_to_end(grid, [i,j], end)
    print("Reward table has been calculated!")
    Actor = actor()
    Critic = critic()
    if actor_model is not None:
        Actor.old_pi.load_state_dict(actor_model)
        Actor.new_pi.load_state_dict(actor_model)
        Critic.get_critic_V.load_state_dict(critic_model)
        save = {'net': Actor.old_pi.state_dict(), 'i': -1}
        torch.save(save, "last_trained_model\Actor_Model.pth")
        save = {'net': Critic.get_critic_V.state_dict(), 'i': -1}
        torch.save(save, "last_trained_model\Critic_Model.pth")

    for ep in range(EP_MAX):
        goal = False
        samples = []
        #memory = []
        states_env = []
        pi_old = []
        pi_new = []
        actions = []
        now = copy.deepcopy(start)
        #memory.append(tuple(now))
        this_ep_reward = 0
        Return = []
        k = 0
        viewed_times = {}
        length = len(grid)
        width = len(grid[0])
        for i in range(length):
            for j in range(width):
                if grid[i][j] == 0:
                    viewed_times[tuple((i,j))] = 0
        viewed_times[tuple(now)] = 1
        for _ in range(SAMPLE_NUM): # record N x samples[st, at, rt, st+1]
            #if len(memory) > memory_size:
            #    memory.pop(0)
            state_env = get_state(grid, now, end)
            state_env = torch.tensor(state_env,dtype=torch.float32)
            states_env.append(state_env)
            this_pi_old = Actor.old_pi.forward(state_env)
            pi_old.append(this_pi_old)
            pi_new.append(Actor.new_pi.forward(state_env))
            action = choose_action(this_pi_old)
            actions.append(action)
            next = do_action(grid, now, action)
            viewed_times[tuple(next)] = viewed_times[tuple(next)] + 1
            #memory.append(tuple(next))
            #if len(memory) > memory_size:
            #    memory.pop(0)
            reward = get_reward(now, next, start, end, viewed_times, _)

            this_ep_reward = GAMA * this_ep_reward + reward
            Return.append(this_ep_reward)
            scaled_reward = np.clip((reward/(np.array(Return).std() + 1e-7)), -10, 10)
            next_state = get_state(grid, next, end)
            samples.append([now, state_env, action, scaled_reward, next, next_state])
            if next == end:
                goal = True
                break
            if now != next:
                now = copy.deepcopy(next)
            else:
                k = k + 1
                if k == 4096:
                    goal = False   # reduce suffering
                    break

        td_e = Critic.learn(samples, ep)
        Actor.learn(td_e, pi_old, pi_new, actions, states_env)
        Actor.update_old()
        writer.add_scalar('Done or not, 1 is done', goal, ep)
        writer.add_scalar("trajectory rewards", this_ep_reward, ep)
        writer.add_scalar('This ep time steps', SAMPLE_NUM, ep)
        if ep % 10 == 0:
            save = {'net': Actor.old_pi.state_dict(), 'i': ep}
            torch.save(save, "model\Actor_Model.pth")
            save = {'net': Critic.get_critic_V.state_dict(), 'i': ep}
            torch.save(save, "model\Critic_Model.pth")
            #print('ep =', ep, ', reward =', this_ep_reward, 'Found goal?', goal)
    return length_to_end



