from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, D_in, H, D_out, dropout=0.5, initialize=None):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(D_in, H, bias=False),
            nn.BatchNorm1d(H, affine=True),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(D_in, H, bias=False),
            nn.BatchNorm1d(H, affine=True),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(H, D_out)
        )

        if not initialize is None:
            initialize(self)

    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        y = self.classifier(h1 + h2)
        return y

class MLP_vanilla(nn.Module):
    def __init__(self, D_in, H, D_out, dropout=0.5, initialize=None):
        super(MLP_vanilla, self).__init__()

        self.fc1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(H, D_out)

        if not initialize is None:
            initialize(self)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
