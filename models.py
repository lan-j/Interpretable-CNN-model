from torch import nn
import torch


class TCN(nn.Module):
    def __init__(self, length, channel_size, output_size, kernel_size, hidden_size, linear_size):
        super(TCN, self).__init__()
        self.conv = nn.Conv1d(in_channels=length, out_channels=channel_size, kernel_size=kernel_size, bias=False)
        self.relu = nn.ReLU()
        # self.hidden_size = 0
        # self.maxpool = nn.AdaptiveMaxPool1d(output_size)
        # self.lstm = nn.LSTM(channel_size, hidden_size, 2, batch_first=True)
        # self.dropout = nn.Dropout(0)
        # self.fc1 = nn.Linear(linear_size, 600)
        self.fc2 = nn.Linear(channel_size * 3, length)
        # self.fc3 = nn.Linear(channel_size, length)
        # self.fc4 = nn.Linear(600, length)
        # self.softmax = nn.Softmax()
        self.sig = nn.Sigmoid()
        # self.sequence = nn.Sequential(self.conv)
        # self.net = nn.Sequential(self.fc1, self.relu,
        #                          self.dropout, self.fc2, self.relu)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0.5, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        # self.fc1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.conv(x.transpose(1, 2))
        # y1 = self.relu(y1)
        y2 = torch.flatten(y1, start_dim=1)

        # y3 = self.dropout(y2)
        # y4 = self.fc1(y3)
        # y5 = self.sig(y4)

        # self.hidden_size = y2.shape[1]
        # y3 = self.fc1(y2)
        # y2 = self.maxpool(y1)
        #
        # y3, _ = self.lstm(y2.transpose(1, 2))
        # y3 = y3[:, -1, :]
        y3 = torch.squeeze(y2)
        y3 = self.fc2(y3)
        # y5 = self.fc4(y4)
        # y5 = self.sig(y5)

        return y3

    def regularization_multi(self):
        # (1- ∑w) + L1
        # r_s = torch.sum(torch.pow(torch.abs(1 - torch.sum(torch.pow(self.conv.weight, 3), dim=1)), 2))
        # L1 = torch.sum(torch.abs(self.conv.weight))
        L1 = self.conv.weight
        L2 = torch.pow(self.conv.weight, 2)
        # print(r_s.item(), L1.item())
        regular = torch.sum(torch.abs(L2-L1))
        # print(lasso)
        # print(L1)
        return regular

    def regularization_uni(self):
        # (1- ∑w) + L1
        r_s = torch.sum(torch.pow(1 - torch.sum(torch.pow(self.conv.weight, 3), dim=1), 2))
        L1 = torch.sum(self.conv.weight)

        # print(r_s.item(), L1.item())
        # print(lasso)
        # print(L1)
        return r_s, L1