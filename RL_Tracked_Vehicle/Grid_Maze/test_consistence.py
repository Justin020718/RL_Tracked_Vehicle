import PPO_agent
import torch
import copy
import A_star_algorithm
Actor_Path = 'model/Actor_Model.pth'

def solve(grid, start, end):
    #reward = 0
    flag = False
    step = 1
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
    action_done = {}
    action_done['up'] = 0
    action_done['down'] = 0
    action_done['left'] = 0
    action_done['right'] = 0
    for i in range(step):
        pi = Actor.new_pi.forward(torch.tensor(state, dtype=torch.float32))
        action = PPO_agent.choose_action(pi)
        action_done[action] = action_done[action] + 1
        next = PPO_agent.do_action(grid, pos, action)
        while next == pos:
            action = PPO_agent.choose_action(pi)
            action_done[action] = action_done[action] + 1
            next = PPO_agent.do_action(grid, pos, action)
    return True, 'Test'