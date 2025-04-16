# coding:utf-8
import time
import numpy as np
import torch
from data_make import real_test, test_sim, pre_allin, real_train
from train_data import train_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load("sim.pth" , map_location='cpu' ).to(device)


def cho_sup(n):
    org_data, _ = real_train()
    s0 = org_data[:12]
    s1 = org_data[12:24]
    s2 = org_data[24:36]
    s3 = org_data[36:48]
    s4 = org_data[48:60]
    s5 = org_data[60:72]
    s6 = org_data[72:84]
    s7 = org_data[84:96]
    return s0[:n], s1[:n], s2[:n], s3[:n], s4[:n], s5[:n], s6[:n], s7[:n]


def choose_test(num):
    test_data, test_label = real_test()
    return test_data[num:num+1], test_label[num:num+1]


# s0,s1,s2,s3,s4,s5,s6,s7=slice_sup()
s0, s1, s2, s3, s4, s5, s6, s7 = cho_sup(1)

te_data, te_label = real_test()

a = 0
t=0
for ln in range(8):
    lnn = ln*60
    test_data = te_data[lnn:lnn+60]
    test_label = te_label[lnn:lnn+60]

    time_start = time.time()
    p0 = test_sim(s0, test_data, model, device)
    p1 = test_sim(s1, test_data, model, device)
    p2 = test_sim(s2, test_data, model, device)
    p3 = test_sim(s3, test_data, model, device)
    p4 = test_sim(s4, test_data, model, device)
    p5 = test_sim(s5, test_data, model, device)
    p6 = test_sim(s6, test_data, model, device)
    p7 = test_sim(s7, test_data, model, device)

    index = pre_allin(p0, p1, p2, p3, p4, p5, p6, p7)
    test_label = test_label.reshape([-1, 1])
    acc = np.concatenate((index, test_label), axis=1)

    for i in range(acc.shape[0]):
        if acc[i, 0] == acc[i, 1]:
            a += 1
    time_end = time.time()
    t +=(time_end-time_start)

print( (a / 480) * 100 )
print(t)



