import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(kernel_size=2)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.Output = nn.Linear(in_features=84, out_features=12)

    def forward(self, x):
        x = self.C1(x)
        x = self.relu1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.relu2(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.F6(x)
        x = self.relu4(x)
        x = self.Output(x)

        return x

if __name__ == '__main__':
    x = torch.randn(1, 1, 28, 28)
    model = LeNet5()
    print(model)
