## derived from Udacity DDPG workbook

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    dim = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(dim)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, random_seed=42, fc1_units = 256, fc2_units = 256):
        """Initialize parameters and model architecture
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            random_seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer        
        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(random_seed)
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, 0, self.fc1_units**-0.5)
        nn.init.normal_(self.fc2.weight, 0, self.fc2_units**-0.5)
        nn.init.xavier_uniform_(self.fc3.weight)
    
class Critic(nn.Module):
    """Critic Model estimates the value (Q)."""
    
    def __init__(self, state_size, action_size, random_seed=41, fc1_units=128, fc2_units=128):
        """Initialize parameters and build the model
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            random_seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, 0, self.fc1_units**-0.5)
        nn.init.normal_(self.fc2.weight, 0, self.fc2_units**-0.5)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state, action):
        """Critic (Value) network that maps (state, action) pairs to Q-values."""
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1) # combine the neural activations of the first hidden layer and our actions
        x = F.relu(self.fc2(x))
        return self.fc3(x)