import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128,128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QNetworkContinuous(nn.Module):
    """
    Q-network for continuous actions: takes state and action concatenated -> scalar Q.
    """
    def __init__(self, state_dim, action_dim=1, hidden_dims=(128,128)):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        # s: [B, state_dim], a: [B, action_dim] or [B,1]
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)
