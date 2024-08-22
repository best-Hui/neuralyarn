import torch
import torch.nn as nn

class Model_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.sigmoid
        self.sequential = nn.Sequential(
            nn.Linear(3, 7), nn.PReLU(7),
            nn.Linear(7, 7), nn.PReLU(7),
            nn.Linear(7, 1)
        )

    def forward(self, x):
        return self.activation(self.sequential(x))

class Model_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.exp
        self.sequential = nn.Sequential(
            nn.Linear( 6, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21,  3)
        )

    def forward(self, x):
        return self.activation(self.sequential(x))