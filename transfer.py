import torch.nn as nn
from random import shuffle, sample
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from data import to_onehot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pearsonr_ci import pearsonr_ci
from data import align


def plot(model, args):
    for param in model.parameters():
        param.requires_grad = False
    new_model = list(model.children())[0]

    if args.multi_pattern:
        x_label_list_multi = ['exam', 'forumng', 'gap', 'homepage', 'other', 'oucontent', 'ouwiki', 'quiz',
                              'register', 'resource', 'subpage', 'transfer', 'unregister', 'url']
    else:
        x_label_list_uni = ['aulaweb', 'blank', 'deeds', 'diagram', 'fsm', 'other',
                            'properties', 'study', 'texteditor']
    # x_label_list_uni_sample = ['A', 'B', 'C', 'D', 'E', 'F',
    #                            'G', 'H', 'I']
    y_label = ['T1', 'T2', 'T3']
    act = new_model.state_dict()['weight'].transpose(1, 2)
    fig = plt.figure(act.size(0), figsize=(8, 4.5))
    for idx in range(11, 15):
        fig.add_subplot(2, 2, idx-10)
        plt.imshow(act[idx - 1], cmap='binary',  interpolation='nearest')

        plt.yticks(range(len(y_label)), y_label)
        if idx > 12:
            plt.xticks(range(len(x_label_list_uni)), x_label_list_uni, rotation='vertical')
    fig.suptitle('The following four rows are patterns learned by our method. \n The top two rows are learned on EPM; the others are learned on OULAD.', fontsize=16)
    plt.xlabel(args.course + str(args.channel_size))
    plt.show()


def transfer(data, model, args):
    for param in model.parameters():
        param.requires_grad = False
    new_model = list(model.children())[0]
    act = new_model.state_dict()['weight'].transpose(1, 2)

    # # baseline
    # new_model.weight = nn.Parameter(torch.zeros_like(new_model.weight), requires_grad=False)
    # action_index = []
    # action_support = []
    # f = open('./baseline/'+args.course+'_seq', 'r')
    # for line in f:
    #     action_index.append(line.split(' -1 ')[:-1])
    #     action_support.append(int(line.split(' -1 ')[-1].split(':')[-1]))
    # zipped = list(zip(action_index, action_support))
    # res = sorted(zipped, key=lambda x: -x[1])
    # action_index = [i for i, j in res[:25]]
    # # random.seed(0)
    # # action_index = sample(action_index, 25)
    #
    # for i in range(len(action_index)):
    #     for j in range(len(action_index[i])):
    #         new_model.weight[i, int(action_index[i][j]), j] = 1.
    #

    r_s = torch.abs(torch.pow(1 - torch.sum(torch.pow(act, 3), dim=2), 3))
    L1 = torch.sum(torch.abs(act))

    total = []
    for stu_data in data:
        test_x = to_onehot(stu_data['seq']).transpose(0, 1).unsqueeze(0)
        out = torch.squeeze(new_model(test_x))
        list_out = [stu_data['id']]

        num_out = np.concatenate((
            out.sum(dim=1).numpy(), out.std(dim=1).numpy(),
            out.max(dim=1)[0].numpy(), out.min(dim=1)[0].numpy(),
            skew(out, axis=1), kurtosis(out, axis=1),
            np.quantile(out, 0.1, axis=1), np.quantile(out, 0.3, axis=1),
            np.quantile(out, 0.5, axis=1), np.quantile(out, 0.7, axis=1), np.quantile(out, 0.9, axis=1)
        ), axis=0).tolist()

        list_out.extend(num_out)
        total.append(list_out)
    total = pd.DataFrame(total).set_index(0)
    test_y = pd.read_csv('data/'+args.course+'_grade.csv', header=None, index_col=0)
    test = total.merge(test_y, left_index=True, right_index=True)

    test_y = test['1_y']
    test_x = test.drop(['1_y'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(test_x, test_y, test_size=0.33, random_state=20)
    lr = RandomForestRegressor(25, random_state=66)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    cor = np.corrcoef(y_test, y_pred)[0][1]
    print(cor)
    r, p, lo, hi = pearsonr_ci(x=y_test, y=y_pred)
    print(r, p, lo, hi)
