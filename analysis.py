import pandas as pd
import matplotlib.pyplot as plt
import json

file = pd.read_csv('mean_crossstu_A.csv', header=0, index_col=0)
file = file.sort_values(by=['1_x'], ascending=False)
print(file)
train_Y = file['1_y']
# print(file[:60])
# print(o)
# print(o)
train_X = file.drop(['1_y'], axis=1)
print(train_X['1_x'].describe())
plt.plot(list(train_X['1_x']), list(train_Y))
plt.xlabel('pattern matching')
plt.ylabel('grade')
plt.show()
print(o)
file = pd.concat([train_Y, train_X.sum(axis=1)], axis=1, join='inner')
file.describe()
x_label_list_multi = ['bos', 'exam', 'forumng', 'gap', 'homepage', 'other', 'oucontent', 'ouwiki', 'quiz', 'register',
                      'resource', 'subpage', 'transfer', 'unregister', 'url', 'eos']

with open('/Users/jianglan/sequence_data/anomaly/data/A.json') as json_file:
    data = json.load(json_file)
    convert = []
    for i in data['2552472_2013J']:
        convert.append([x_label_list_multi[t] for t in i])
    print(convert)