import json
import torch


def load(course):
    with open('data/'+course+'.json', "r") as content:
        j = json.load(content)
        data = []
        for i in j:
            s = [[], [], [], []]
            norm = []
            for k in j[i]:
                norm.append([l-1 for l in k])
            s.extend(norm)
            data.append({'seq': s, 'id': i})
    return data


def breakintosubpattern(data, k):
    patterns = []
    for i in data:
        for j in range(len(i['seq'])-k-1):
            patterns.append([i['seq'][j:j+k], i['seq'][j+k+1]])
    return patterns


def align(data):
    num = [len(e['seq']) for e in data]
    max_num = max(num)
    id = [int(e['id']) for e in data]
    batch_seq = []
    for i in data:
        s_ = i['seq'].copy()
        s_.extend([[]]*(max_num-len(s_)))
        batch_seq.append(to_onehot(s_))
    batch_seq = torch.stack(batch_seq, dim=0)
    return batch_seq, num, id


def to_onehot(labels, n_categories=9, dtype=torch.float32):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
    for i, label in enumerate(labels):
        label = torch.LongTensor(label)
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
    return one_hot_labels


def colloate_fn(batch):
    batch_seq = []
    batch_truth = []
    for i in batch:
        s_ = i[0].copy()
        batch_seq.append(to_onehot(s_))
        s_t = i[1].copy()
        batch_truth.append(to_onehot([s_t]))
    batch_seq = torch.stack(batch_seq, dim=0)
    batch_truth = torch.stack(batch_truth, dim=0)
    return batch_seq, batch_truth
