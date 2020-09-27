import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from predictive.data import load, breakintosubpattern, colloate_fn
from predictive.models import TCN
from predictive.transfer import transfer
import statistics as stat
from random import shuffle

from sklearn.model_selection import train_test_split

import visdom
from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import csv

torch.set_printoptions(threshold=5000)

parser = argparse.ArgumentParser(description='TCN')

# Model parameters
parser.add_argument('--lamda1', type=float, default=0.075,
                    help='strength of sum part or the regularization')
parser.add_argument('--lamda2', type=int, default=0.00075,
                    help='strength of L1')
parser.add_argument('--current_event', type=int, default=5)
parser.add_argument('--channel_size', type=int, default=25)
parser.add_argument('--kernal_size', type=int, default=3)

# learning parameter
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--warm_up', type=bool, default=1)
parser.add_argument('--warm_up_epoch', type=int, default=5)
parser.add_argument('--multi_pattern', type=bool, default=1)
parser.add_argument('--course', default='G')
parser.add_argument('--l_s', type=bool, default=0)

title = 'Rgl_wmup'

args = parser.parse_args()
torch.set_printoptions(edgeitems=4)
# print(args)
data = load(args.course)
data_ = breakintosubpattern(data, args.current_event)
train_set, test_set = train_test_split(data_, test_size=0.1, random_state=28)
train_num = len(train_set)
test_num = len(test_set)
print('Data Loaded Complete')
print('==========================================================')


# all_actions = []
# for stu in data:
#     actions = []
#     for action in stu:
#         ind = np.where(action == 1)
#         actions.append(' '.join(list(map(str, list(ind[0])))))
#     all_actions.append(actions)
# with open(args.course + '.txt', 'w') as file:
#     for line in all_actions:
#         s = " -1 ".join(line)
#         s = s + " -2"
#         file.write('%s\n' % s)
# print(o)
# print('==========================================================')

# if args.l_s:
#     shuffle(data)
c_event = args.current_event
f_event = 1
batch_size = args.batch_size

# model configuration

if args.l_s:
    channel_sizes = args.channel_size
    kernel_size = args.kernal_size
    epochs = args.num_epochs
    hidden_size = 40
    length = 9
    if args.multi_pattern:
        length = 14
    linear_size = channel_sizes * (c_event - kernel_size + 1)
    model = TCN(length, channel_sizes, f_event, kernel_size, hidden_size, linear_size)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
# print(model.conv.weight)
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
# print(o)

# weight = torch.FloatTensor([1, 2, 3, 5, 5, 5, 1, 1])


# criterion = nn.MultiLabelSoftMarginLoss()


# experiment

loss_values_train = []
loss_values_test = []


def train(ep, lambda1, lambda2, args):
    # print(lambda1, lambda2)
    global batch_size, iters, epochs
    print('Epoch: {}'.format(ep))
    total_loss = 0
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    # training set
    progress = tqdm.tqdm(total=train_num/32, ncols=75, desc='Train {}'.format(ep))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=True, drop_last=True, collate_fn=colloate_fn)):
        predict = model(batch[0])
        loss = criterion(predict, torch.squeeze(batch[1]))
        total_loss += loss
        regularization = 0
        if 'Rgl' in title:
            if args.multi_pattern:
                regularization = lambda1 * model.regularization_multi()
            else:
                r_s, L1 = model.regularization_uni()
                regularization = lambda1 * r_s + lambda2 * L1

        loss = loss + regularization
        loss.backward()

        progress.update(1)
        optimizer.step()
        optimizer.zero_grad()
    progress.close()
    return total_loss, predict


def evaluate():
    model.eval()
    total_loss = 0
    progress = tqdm.tqdm(total=test_num/32, ncols=75, desc='Test {}'.format(ep))
    for batch in DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=colloate_fn):
        progress.update(1)
        predict = model(batch[0])
        loss = criterion(predict, torch.squeeze(batch[1]))
        total_loss += loss
    progress.close()
    return total_loss, predict


if args.l_s:
    best = 1000
    liveloss = PlotLosses()
    logs = {}
    # vis = visdom.Visdom()
    # # vis_test = visdom.Visdom()
    # train_loss_window = vis.line(X=np.zeros((1,)),
    #                        Y=np.zeros((1,)),
    #                        opts=dict(xlabel='epoch',
    #                                  ylabel='Train_Loss',
    #                                  title=title,
    #                                  ))
    # test_loss_window = vis.line(X=np.zeros((1,)),
    #                        Y=np.zeros((1,)),
    #                        opts=dict(xlabel='epoch',
    #                                  ylabel='Test_Loss',
    #                                  title=title,
    #                                  ))
    for ep in range(args.num_epochs):
        # loss_train, predict_train = train(ep, 0, 0, args)
        if args.warm_up:
            if ep < args.warm_up_epoch:
                loss_train, predict_train = train(ep, 0, 0, args)
            elif ep < args.warm_up_epoch + 10 and ep >= args.warm_up_epoch:
                l1 = (ep - args.warm_up_epoch) * args.lamda1 / 10
                l2 = (ep - args.warm_up_epoch) * args.lamda2 / 10
                loss_train, predict_train = train(ep, l1, l2, args)
            else:
                loss_train, predict_train = train(ep, args.lamda1, args.lamda2, args)
        else:
            loss_train, predict_train = train(ep, args.lamda1, args.lamda2, args)
        epoch_train_loss = loss_train.detach() / train_num
        logs['loss'] = epoch_train_loss.item()

        # loss_train, predict_train = train(ep, args.lamda1, args.lamda2, args)
        loss_test, predict_loss = evaluate()
        epoch_test_loss = loss_test.detach() / test_num

        logs['val_loss'] = epoch_test_loss.item()
        # transfer(data, model, args)
        liveloss.update(logs)
        liveloss.send()
        # if loss_train <= best:
        #     best = loss_train
        # transfer(data, model, args)
    torch.save(model, 'model/model_' + args.course + '_' + title + '.pt')
        #     print('Train Loss: ', loss_train / train_num)
        #     print('Test Loss: ', loss_test/test_num)

        # X1 = np.ones((1, 1)) * ep
        # Y1 = np.array([loss_train.detach()/train_num])
        # Y2 = np.array([loss_test.detach()/test_num])
        # vis.line(
        #     X=np.column_stack(X1),
        #     Y=np.column_stack(Y1),
        #     win=train_loss_window,
        #     update='append')
        # vis.line(
        #     X=np.column_stack(X1),
        #     Y=np.column_stack(Y2),
        #     win=test_loss_window,
        #     update='append')

model = torch.load('model/model_' + args.course + '_' + title +'.pt')
transfer(data, model, args)

# if args.l_s:
#     for ep in range(args.num_epochs):
#         # train(ep, 0, 0, args)
#         if ep <= 10:
#             train(ep, 0, 0, args)
#         elif ep < args.num_epochs:
#             if args.warm_up:
#                 if ep <= args.warm_up_epoch + 10:
#                     l1 = (ep - 10) * args.lamda1 / args.warm_up_epoch
#                     l2 = (ep - 10) * args.lamda2 / args.warm_up_epoch
#                     train(ep, l1, l2, args)
#                 else:
#                     train(ep, args.lamda1, args.lamda2, args)
#             else:
#                 train(ep, args.lamda1, args.lamda2, args)
#         evaluate()
#     print(args.course + str(args.channel_size))
#     torch.save(model, 'model/model-'+args.course + str(args.channel_size)+'.pt')
# else:
#     model = torch.load('model/model-' + args.course + str(args.channel_size) + '.pt')
#     transfer(model)


# if args.l_s:
#     print('Loss of Train', loss_values_train)
#     print('Loss of Test', loss_values_test)
#
#     with open('./model/'+args.course+str(args.channel_size)+'.txt', 'w') as f:
#         f.write('Loss of Train\n')
#         for item in loss_values_train:
#             f.write("%s\n" % item)
#         f.write('Loss of Test\n')
#         for item in loss_values_test:
#             f.write("%s\n" % item)
#
#     r_sum = []
#     r_l1 = []
#     l_train = []
#     c = 0
#     for i in loss_values_train:
#         r_sum.append(i[1])
#         # r_l1.append(i[2])
#         l_train. append(i[0])
#         c += 1
#     # plt.plot(np.array(r_l1), 'r')
#     # # plt.plot(np.array(loss_values_train), 'b')
#     # plt.show()
#     # plt.plot(np.array(r_sum), 'b')
#     # plt.show()
#     plt.plot(np.array(loss_values_test), 'r')
#     plt.plot(np.array(l_train), 'g')
#     plt.show()


print(args)