# coding:utf-8
import random
import numpy as np
import torch
from torch import where
from torch.utils.data import TensorDataset

def s_100(data):
    num=data.shape[0]
    d = data.reshape([ num, 80, 6 ])
    return d



def create_10s(x, digit_indices):  # 构造对，交替生成正负样本对，输入x图像数据    及每类的索引标签
    pairs = []  # 存储样本对
    labels = []  # 存储样本对对应的标签
    n = min([len(digit_indices[d]) for d in range(8)])  # 中括号里面计算每一类的样本数，n是最小的样本数-1
    for d in range(8):  # 对每一类进行操作
        for i in range(n):
            inc = random.randrange(1, 3)     #随机数
            #inc = 1
            i1 = (i + inc) % n
            i2 = (i + 2*inc) % n
            i3 = (i + 3*inc) % n
            i4 = (i + 4*inc) % n

            z11, z12 = digit_indices[d][i], digit_indices[d][i1]  #这是寻找同类图片的索引
            z21, z22 = digit_indices[d][i], digit_indices[d][i2]
            z31, z32 = digit_indices[d][i], digit_indices[d][i3]
            z41, z42 = digit_indices[d][i], digit_indices[d][i4]

            pairs += [[x[z11], x[z12]]]  # 获得每张图的标签后，用索引找到位置，然后和同类构成正样本对标签为1
            pairs += [[x[z21], x[z22]]]
            pairs += [[x[z31], x[z32]]]
            pairs += [[x[z41], x[z42]]]

            dn1 = (d + 1) % 8          #得到除法的余数
            dn2 = (d + 2) % 8
            dn3 = (d + 3) % 8
            dn4 = (d + 4) % 8

            a11, a12 = digit_indices[d][i], digit_indices[dn1][i]  #寻找不同类
            a21, a22 = digit_indices[d][i], digit_indices[dn2][i]
            a31, a32 = digit_indices[d][i], digit_indices[dn3][i]
            a41, a42 = digit_indices[d][i], digit_indices[dn4][i]

            pairs += [[x[a11], x[a12]]]  # 和不同的类构成负样本对标签为0
            pairs += [[x[a21], x[a22]]]
            pairs += [[x[a31], x[a32]]]
            pairs += [[x[a41], x[a42]]]

            labels += [1,1,1,1,
                       0,0,0,0]
    return np.array(pairs), np.array(labels)  # 返回样本对和标签


def mat_train():
    d0 = np.loadtxt('data_mat/train.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data_mat/tr_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all



def hunhe_train():
    d0 = np.loadtxt('TTS_data/train.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('TTS_data/tr_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all

def real_train():
    d0 = np.loadtxt('data/data/train.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/tr_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all

def real_many():
    d0 = np.loadtxt('data/data/many.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/ma_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all


def train_r(e):
    en=e * 12
    d0 = np.loadtxt('data/data/train.csv', delimiter=',')[ en : en + 12 ]
    d0 = s_100(d0)
    data = torch.FloatTensor(d0)
    return data


def train_many(e):
    en=e * 120
    d0 = np.loadtxt('data/data/many.csv', delimiter=',')[ en : en + 120 ]
    d0 = s_100(d0)
    data = torch.FloatTensor(d0)
    return data


def real_val():
    d0 = np.loadtxt('data/data/val.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/va_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all


def real_test():
    d0 = np.loadtxt('data/data/test.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/test_la.csv', delimiter=',')
    label = label.reshape(-1)
    label = torch.LongTensor(label)
    data = torch.FloatTensor(d0)
    all = TensorDataset(data, label)
    return all



def test_sim(support,data,model ,device):  # 将每个支持数据和故障数据合成一对
    pairs = []  # 存储样本对
    for d in range(support.shape[0]):  # 样本集的每个样本
        for i in range(data.shape[0]):  #支持集的每个样本
            pairs += [[support[d], data[i]]]  #一个支持样本配一堆
    test = torch.Tensor(pairs)
    pre = model( test[:, 0].to( device ), test[:, 1].to( device ) )   #输出成1列
    pre_all = pre.view([support.shape[0], data.shape[0]])
    pre_all = pre_all.T.cpu().detach().numpy()
    pre_mean=np.mean(pre_all, axis=1)
    pre_mean=pre_mean.reshape([-1,1])

    return np.array(pre_mean)   #行为与支持的相似度，列为数据样本



def pre_allin(a, b, c,d,e,f,g,h):  #随机化
    pre = np.concatenate((a, b, c,d,e,f,g,h), axis=1)
    index= np.argmin(pre, axis = 1).reshape([-1,1])

    return index