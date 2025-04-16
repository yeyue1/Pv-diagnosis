# coding:utf-8
import numpy as np
import pandas as pd

def data_4(l):
    lis = []
    for i in range( int( l.shape[0]//80 ) ):
        leg = i * 80
        da = l[leg: leg + 80]
        dat = da.reshape([ 1, -1 ])
        lis.append(dat)
    ld = np.array( lis ).reshape( [ -1, 480 ] )

    rng = np.random.default_rng()
    data0 = rng.permutation(ld, axis=0)
    dat_e=data0[:12]     # 0.05 , 7个
    dat_k=data0[12:20]
    dat_r=data0[20:80]
    dat_m=data0[80:200]
    return dat_e,dat_k,dat_r,dat_m

def t_14():
    n = 'nor_data'
    d0 = np.loadtxt(n + '/0.csv', delimiter=',')
    d1 = np.loadtxt(n + '/1.csv', delimiter=',')
    d2 = np.loadtxt(n + '/2.csv', delimiter=',')
    d3 = np.loadtxt(n + '/3.csv', delimiter=',')
    d4 = np.loadtxt(n + '/4.csv', delimiter=',')
    d5 = np.loadtxt(n + '/5.csv', delimiter=',')
    d6 = np.loadtxt(n + '/6.csv', delimiter=',')
    d7 = np.loadtxt(n + '/7.csv', delimiter=',')
    d0, t0, r0, s0 = data_4(d0)
    d1, t1, r1, s1 = data_4(d1)
    d2, t2, r2, s2 = data_4(d2)
    d3, t3, r3, s3 = data_4(d3)
    d4, t4, r4, s4 = data_4(d4)
    d5, t5, r5, s5 = data_4(d5)
    d6, t6, r6, s6 = data_4(d6)
    d7, t7, r7, s7 = data_4(d7)
    label = np.loadtxt('nor_data/label.csv', delimiter=',')

    l=label[:d0.shape[0]].T
    d = np.concatenate((d0, d1, d2, d3, d4, d5,d6,d7), axis=0)
    l = l.reshape(-1)

    t = label[:t0.shape[0]].T
    dt = np.concatenate((t0,t1,t2,t3,t4,t5,t6,t7), axis=0)
    t = t.reshape(-1)

    s = label[:r0.shape[0]].T
    dr = np.concatenate((r0, r1, r2, r3, r4,r5,r6,r7), axis=0)
    s = s.reshape(-1)

    a_0=label[:s0.shape[0]].T
    a_s=np.concatenate((s0, s1, s2, s3, s4,s5,s6,s7), axis=0)
    a_0 = a_0.reshape(-1)

    return d,l,dt,t,dr,s,a_s,a_0


data, label, dt, t, dr, s, a_s, a_0 = t_14()

data=pd.DataFrame(data.reshape(data.shape[0],-1))
label=pd.DataFrame(label)

k1=pd.DataFrame(dt.reshape(dt.shape[0],-1))
k2=pd.DataFrame(t)

s1=pd.DataFrame(dr.reshape(dr.shape[0],-1))
s2=pd.DataFrame(s)

a_s1=pd.DataFrame(a_s.reshape(a_s.shape[0],-1))
a_0=pd.DataFrame(a_0)

data.to_csv('data/train.csv',header=False,index=False)
label.to_csv('tr_la.csv',header=False,index=False)

k1.to_csv('data/val.csv',header=False,index=False)
k2.to_csv('va_la.csv',header=False,index=False)

s1.to_csv('data/test.csv',header=False,index=False)
s2.to_csv('test_la.csv',header=False,index=False)

a_s1.to_csv('data/many.csv',header=False,index=False)
a_0.to_csv('ma_la.csv',header=False,index=False)
