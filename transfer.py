import torch.nn as nn
from random import shuffle, sample
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from predictive.data import to_onehot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from predictive.pearsonr_ci import pearsonr_ci
from predictive.data import align


def transfer(data, model, args):
    # print(model.state_dict())
    for param in model.parameters():
        param.requires_grad = False
    new_model = list(model.children())[0]

    # baseline
    new_model.weight = nn.Parameter(torch.zeros_like(new_model.weight), requires_grad=False)
    action_index = []
    action_support = []
    f = open('./baseline/'+args.course+'_seq', 'r')
    for line in f:
        action_index.append(line.split(' -1 ')[:-1])
        action_support.append(int(line.split(' -1 ')[-1].split(':')[-1]))
    zipped = list(zip(action_index, action_support))
    res = sorted(zipped, key=lambda x: -x[1])
    action_index = [i for i, j in res[:25]]
    # random.seed(0)
    # action_index = sample(action_index, 25)

    for i in range(len(action_index)):
        for j in range(len(action_index[i])):
            new_model.weight[i, int(action_index[i][j]), j] = 1.
    #
    # # # #
    # # plot
    # x_label_list_multi = ['exam', 'forumng', 'gap', 'homepage', 'other', 'oucontent', 'ouwiki', 'quiz',
    #                       'register', 'resource', 'subpage', 'transfer', 'unregister', 'url']
    # # x_label_list_multi = ['reg', 'unreg', 'trans', 'forumn', 'ouc', 'subpage', 'homepage',
    # #           'quiz', 'resource', 'url', 'ouwiki', 'other', 'exam', 'gap']
    # x_label_list_uni = ['aulaweb', 'blank', 'deeds', 'diagram', 'fsm', 'other',
    #           'properties', 'study', 'texteditor']
    # # x_label_list_uni = ['fsm', 'study', 'blank', 'aulaweb', 'other', 'properties',
    # #           'texteditor', 'deeds', 'diagram']
    # x_label_list_uni_sample = ['A', 'B', 'C', 'D', 'E', 'F',
    #                            'G', 'H', 'I']
    # y_label = ['T1', 'T2', 'T3']
    # act = new_model.state_dict()['weight'].transpose(1, 2)
    # r_s = torch.abs(torch.pow(1 - torch.sum(torch.pow(act, 3), dim=2), 3))
    # L1 = torch.sum(torch.abs(act))
    # # print(r_s, L1)
    # # print(act.shape)
    # # print(o)
    #
    # fig = plt.figure(act.size(0), figsize=(6, 3))
    # for idx in range(11, 15):
    #     fig.add_subplot(2, 2, idx-10)
    #     plt.imshow(act[idx - 1], cmap='binary',  interpolation='nearest')
    #
    #     plt.yticks(range(len(y_label)), y_label)
    #     if idx > 12:
    #         plt.xticks(range(len(x_label_list_uni)), x_label_list_uni, rotation='vertical')
    # plt.xlabel(args.course + str(args.channel_size))
    # # plt.show()
    #
    # # plt.savefig('/Users/jianglan/sequence_data/predictive/plot_of_CNN/'+ args.course+'_baseline.svg')
    # plt.savefig('/Users/jianglan/sequence_data/predictive/plot_of_CNN/'+ args.course+'_with_label.svg')
    # # print(o)



    # data, num, id = align(data)
    # id = pd.Series(id)
    # out = new_model(data.transpose(1, 2))
    # num_out = np.concatenate((
    #     # out.sum(dim=1).sum(dim=0).numpy(),
    #     out.sum(dim=2).numpy(), out.std(dim=2).numpy(),
    #     out.max(dim=2)[0].numpy(), out.min(dim=2)[0].numpy(),
    #     skew(out, axis=2), kurtosis(out, axis=2),
    #     np.quantile(out, 0.1, axis=2), np.quantile(out, 0.3, axis=2),
    #     np.quantile(out, 0.5, axis=2), np.quantile(out, 0.7, axis=2), np.quantile(out, 0.9, axis=2)
    # ), axis=1)
    # total = pd.DataFrame(num_out).set_index(id)

    total = []
    for stu_data in data:
        # print(stu_data['seq'][2:])
        test_x = to_onehot(stu_data['seq']).transpose(0, 1).unsqueeze(0)
        out = torch.squeeze(new_model(test_x))
        # list_out = [int(stu_data['id'])]
        list_out = [stu_data['id']]

        num_out = np.concatenate((
            # out.sum(dim=1).sum(dim=0).numpy(),
            out.sum(dim=1).numpy(), out.std(dim=1).numpy(),
            out.max(dim=1)[0].numpy(), out.min(dim=1)[0].numpy(),
            skew(out, axis=1), kurtosis(out, axis=1),
            np.quantile(out, 0.1, axis=1), np.quantile(out, 0.3, axis=1),
            np.quantile(out, 0.5, axis=1), np.quantile(out, 0.7, axis=1), np.quantile(out, 0.9, axis=1)
        ), axis=0).tolist()

        list_out.extend(num_out)
        total.append(list_out)
    total = pd.DataFrame(total).set_index(0)
    train_y = pd.read_csv('data/'+args.course+'_grade.csv', header=None, index_col=0)

    train = total.merge(train_y, left_index=True, right_index=True)

    train_Y = train['1_y']
    train_X = train.drop(['1_y'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.33, random_state=20)
    lr = RandomForestRegressor(25, random_state=66)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    cor = np.corrcoef(y_test, y_pred)[0][1]
    print(cor)
    n_test = len(y_test)
    # r, p, lo, hi = pearsonr_ci(n_test=n_test, r=.45)
    # print(r, p, lo, hi)
    r, p, lo, hi = pearsonr_ci(x=y_test, y=y_pred)
    print(r, p, lo, hi)
