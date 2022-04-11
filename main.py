import torch.nn as nn
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from data import load, breakintosubpattern, colloate_fn
from models import TCN
from transfer import transfer, plot

from sklearn.model_selection import train_test_split
import argparse


torch.set_printoptions(threshold=5000)

parser = argparse.ArgumentParser(description='TCN')

# Model parameters
parser.add_argument('--regularization', type=int, default=1,
                    help='strength of L1')
parser.add_argument('--lamda1', type=float, default=0.075,
                    help='strength of sum part or the regularization')
parser.add_argument('--lamda2', type=float, default=0.00075,
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
parser.add_argument('--multi_pattern', type=bool, default=0)
parser.add_argument('--course', default='2')
parser.add_argument('--mode', type=str, default='train')

file_name = ''

args = parser.parse_args()
data = load(args.course)

train, test = train_test_split(data, test_size=0.1, random_state=28)
train_set = breakintosubpattern(train, args.current_event)
test_set = breakintosubpattern(test, args.current_event)
train_num = len(train_set)
test_num = len(test_set)
print('Data Loaded Complete')
print('==========================================================')

c_event = args.current_event
f_event = 1
batch_size = args.batch_size

# model configuration

if args.mode == 'train':
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


# experiment

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
        if 'Rgl' in file_name:
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


if args.mode == 'train':
    best = 1000
    for ep in range(args.num_epochs):
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

        loss_test, predict_loss = evaluate()

    torch.save(model, 'model/model_' + args.course + '_' + file_name + '.pt')

model = torch.load('model/model_' + args.course + '_' + file_name +'.pt')
transfer(test, model, args)

if args.mode == 'plot':
    model = torch.load('model/model_' + args.course + '_' + file_name + '.pt')
    plot(test, model, args)
