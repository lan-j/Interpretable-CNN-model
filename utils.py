import numpy as np
import torch
import secrets
import random
from collections import Counter
from torch.autograd import Variable


def data_generator(all_data, c_event, f_event, b_size):
    length_encoding = all_data[0].shape[1]
    X = np.empty((0, c_event, length_encoding))
    Y = np.empty((0, f_event, length_encoding))
    cou = []
    c = 0
    while 1:
        data = secrets.choice(all_data)
        length = data.shape[0] - c_event - f_event
        if length < 0:
            continue
        s_i = random.randint(0, length)
        X = np.vstack((X, [data[s_i:s_i+c_event]]))
        Y = np.vstack((Y, [data[s_i+c_event:s_i+c_event+f_event]]))
        a = data[s_i+c_event:s_i+c_event+f_event].tolist()[0].index(1)
        cou.append(a)
        c += 1
        if c == b_size:
            print(Counter(cou))
            train_X = torch.from_numpy(X)
            train_Y = torch.from_numpy(Y)
            return train_X, train_Y


def data_sequence_generator(data, c_event):
    length_encoding = data.shape[1]
    X = np.empty((0, c_event, length_encoding))
    for i in range(len(data)-c_event + 1):
        X = np.vstack((X, [data[i:i+c_event]]))
    train_x = torch.from_numpy(X)
    return train_x


