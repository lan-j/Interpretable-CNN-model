from torch import nn
import torch


class TCN(nn.Module):
    def __init__(self, length, channel_size, output_size, kernel_size, hidden_size, linear_size):
        super(TCN, self).__init__()
        self.conv = nn.Conv1d(in_channels=length, out_channels=channel_size, kernel_size=kernel_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_size * 3, length)
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0.5, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.conv(x.transpose(1, 2))
        y2 = torch.flatten(y1, start_dim=1)
        y3 = torch.squeeze(y2)
        y3 = self.fc2(y3)
        return y3

    def regularization_multi(self):
        L1 = self.conv.weight
        L2 = torch.pow(self.conv.weight, 2)
        regular = torch.sum(torch.abs(L2-L1))
        return regular

    def regularization_uni(self):
        rgl = torch.sum(torch.pow(1 - torch.sum(torch.pow(self.conv.weight, 3), dim=1), 2))
        L1 = torch.sum(self.conv.weight)
        return rgl, L1