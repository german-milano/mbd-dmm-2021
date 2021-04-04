import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNModel(nn.Module):
    
    def __init__(self, env_inputs, n_actions):
        super(DQNModel, self).__init__()
        
        # Hidden layer
        self.hidden = nn.Linear(in_features=env_inputs, out_features=200)
        # Output layer
        self.output = nn.Linear(in_features=200, out_features=n_actions)

        
    def forward(self, env_input):
        result = F.relu(self.hidden(env_input))
        output = self.output(result)

        return output

    
class NFQModel(nn.Module):
    
    def __init__(self, env_inputs, n_actions):
        super(NFQModel, self).__init__()
        
        # Hidden layer
        self.hidden = nn.Linear(in_features=env_inputs, out_features=200)
        # Output layer
        self.output = nn.Linear(in_features=200, out_features=n_actions)

        
    def forward(self, env_input):
        result = F.relu(self.hidden(env_input))
        output = self.output(result)

        return output