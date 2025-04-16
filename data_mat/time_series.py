# coding:utf-8
import numpy as np
import pandas as pd

a = np.loadtxt("nor_data/0.csv", delimiter=',')[:7980]
label = []

for n in range(8):
    data_list=[]

    data_1=np.loadtxt("nor_data/"+str(n)+".csv",delimiter=',')[60:]
    for h in range(0,7940,40):
        sp1=a[h : h+60]
        sp2=data_1[h+60 : h+80]
        f_1=np.concatenate([sp1,sp2],axis=0)
        data_list.append(f_1.reshape([1,-1]))
        label.append(n)

    d_list = data_list[:-5]
    data_l=np.array(d_list).reshape([-1,480])
    data_pd=pd.DataFrame(data_l)
    data_pd.to_csv("org_data/" + str(n)+".csv",header=False,index=False)

