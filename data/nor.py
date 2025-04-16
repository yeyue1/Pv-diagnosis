# coding:utf-8
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from matplotlib import pyplot as plt
from mpmath import ln
from numpy.linalg import inv
from scipy.stats import spearmanr

# 21.6  7.68   18   6.668

"""

21.6  7.68   18   6.668
555	21.59629655 17.9 4.346649965 4.130599689 17.968890965193378
355	21.59938381 17.9 2.904672524 2.790500806 17.185994750228936
135	21.57035492 17.9 1.325814282 1.262525258 16.031480667857142
"""
def v(vv = 18 ):
    v1 = 17.9
    v2 = 17.9
    v3 = 17.9

    g1 = 555
    g2 = 355
    g3 = 135

    t1 = 17.968890965193378
    t2 = 17.185994750228936
    t3 = 16.031480667857142

    W = np.array([[ln(g1 / 1000), (t1 - 25), (t1 - 25) * (g1 / 1000)],
                  [ln(g2 / 1000), (t2 - 25), (t2 - 25) * (g2 / 1000)],
                  [ln(g3 / 1000), (t3 - 25), (t3 - 25) * (g3 / 1000)]], dtype=float)

    result = np.array([ v1 -vv, v2 -vv, v3 -vv])

    W_inv = inv(W)
    vxyz = np.dot(W_inv, result)
    v = list(vxyz)
    print("得到的值：")
    print(v[0])
    print(v[1])
    print(v[2])



"""
21.6  7.68   18   6.668
555	21.59629655 17.9 4.346649965 4.130599689 17.968890965193378
355	21.59938381 17.9 2.904672524 2.790500806 17.185994750228936
135	21.57035492 17.9 1.325814282 1.262525258 16.031480667857142
"""

def I(vv = 6.668 ):
    v1 = 4.130599689
    v2 = 2.790500806
    v3 = 1.262525258

    g1 = 555
    g2 = 355
    g3 = 135

    t1 = 17.968890965193378
    t2 = 17.185994750228936
    t3 = 16.031480667857142

    W = np.array([[ (g1 / 1000)*vv, (t1 - 25), (t1 - 25) * (g1 / 1000)],
                  [ (g2 / 1000)*vv, (t2 - 25), (t2 - 25) * (g2 / 1000)],
                  [ (g3 / 1000)*vv, (t3 - 25), (t3 - 25) * (g3 / 1000)]], dtype=float)

    result = np.array([ v1 , v2 , v3 ])

    W_inv = inv(W)
    vxyz = np.dot(W_inv, result)
    v = list(vxyz)
    print("得到的值：")
    print(v[0])
    print(v[1])
    print(v[2])


def voc_norm(data ,vm=21.6 ,a1= 0.100337575011512 ,a2= -0.022704939180458367 ,a3= 0.0267195751458327 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,1]-a1*float(ln(data[n,0]/1000))-a2*(data[n,5]-25)-a3*(data[n,5]-25)*(data[n,0]/1000)
        num=vn/vm
        list.append(2*num-1)
    return np.array(list)

def vm_norm(data ,vm=18 ,a1= -0.0021569160096018944 ,a2= 0.010740892050504429 ,a3= 0.006598623602183178 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,3]-a1*float(ln(data[n,0]/1000))-a2*(data[n,5]-25)-a3*(data[n,5]-25)*(data[n,0]/1000)
        num = vn / vm
        list.append(2 * num - 1)
    return np.array(list)

def isc_norm(data ,im=7.68 ,b1= 0.9496434242761944 ,b2= -0.036612395773737254 ,b3= -0.01062564464216545 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,2]-b2*(data[n,5]-25)-b3*(data[n,5]-25)*(data[n,0]/1000)
        vnt=vn*1000/(data[n,0]*b1)
        list.append(2*(vnt/im)-1)
    return np.array(list)

def im_norm(data ,im= 6.668 ,b1= 0.8581449848923105 ,b2= -0.028552752370328848 ,b3= -0.1932392092960598 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,4]-b2*( data[n,5] -25)-b3*(data[n,5]-25)*(data[n,0]/1000)
        vnt=vn*1000/(data[n,0]*b1)
        list.append(2*(vnt/im)-1)
    return np.array(list)

def irr_norm(data):
    list=[]
    for n in range(data.shape[0]):
        list.append(2*(data[n,0]/1000)-1)
    return np.array(list)

def ff_norm(data):
    list=[]
    for n in range(data.shape[0]):
        ff=(data[n,3]*data[n,4])/(data[n,1]*data[n,2])
        list.append(2*ff-1)
    return np.array(list)


for lab in range(8):
    data = np.loadtxt('de_noise/'+str(lab)+'.csv', delimiter=',')
    irr=irr_norm(data).reshape([-1, 1])
    voc = voc_norm(data).reshape([-1, 1])
    isc = isc_norm(data).reshape([-1, 1])
    vm = vm_norm(data).reshape([-1, 1])
    im = im_norm(data).reshape([-1, 1])
    ff=ff_norm(data).reshape([-1, 1])

    fin = np.concatenate([irr, voc, isc, vm, im,ff], axis=1)
    fina=pd.DataFrame(fin)
    fina.to_csv('nor_data/'+str(lab)+'.csv', header=False,index=False)







