import torch
import torch.nn as nn

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