import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
        )
        self.block2 = nn.Sequential(
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.BatchNorm1d(36),
        )
        self.block3 = nn.Sequential(
            nn.Linear(36, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
        )
        self.block4 = nn.Sequential(
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
        )

        self.fc = nn.Linear(48, 5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.fc(x)

        return x