# coding:utf-8
import numpy as np
import pandas as pd


def t_14():
    n = 'org_data'
    d0 = np.loadtxt(n + '/0.csv', delimiter=',')
    d1 = np.loadtxt(n + '/1.csv', delimiter=',')
    d2 = np.loadtxt(n + '/2.csv', delimiter=',')
    d3 = np.loadtxt(n + '/3.csv', delimiter=',')
    d4 = np.loadtxt(n + '/4.csv', delimiter=',')
    d5 = np.loadtxt(n + '/5.csv', delimiter=',')
    d6 = np.loadtxt(n + '/6.csv', delimiter=',')
    d7 = np.loadtxt(n + '/7.csv', delimiter=',')
    label = np.loadtxt(n + '/label.csv', delimiter=',')

    l=label[:d0.shape[0]].T
    d = np.concatenate((d0, d1, d2, d3, d4, d5,d6,d7), axis=0)
    l = l.reshape(-1)

    return d,l


data,label=t_14()

data=pd.DataFrame(data.reshape(data.shape[0],-1))
label=pd.DataFrame(label)


data.to_csv('train.csv',header=False,index=False)
label.to_csv('tr_la.csv',header=False,index=False)