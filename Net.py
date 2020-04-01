from torch import nn


class Net(nn.Module):
    def __init__(self, size=100):
        super(Net, self).__init__()
        self.linear = nn.Linear(size, size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x.sum(dim=1)
        return x


class ResnetNet(nn.Module):
    def __init__(self, size=100):
        super(ResnetNet, self).__init__()
        self.linear = nn.Linear(size, size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.linear(input)
        x = self.relu(x)
        x = x + input
        x = x.sum(dim=1)
        return x
