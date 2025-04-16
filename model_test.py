# coding:utf-8
import time

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load("data_mat/m6.pth",map_location='cpu').to(device)
mix_model=torch.load("data_mat/hunhe.pth",map_location='cpu').to(device)

s_inc = torch.load("data_mat/s_inc.pth",map_location='cpu').to(device)
s_vgg = torch.load("data_mat/s_vgg.pth",map_location='cpu').to(device)

f_inc = torch.load("data_mat/f_inc.pth",map_location='cpu').to(device)
f_vgg = torch.load("data_mat/f_vgg.pth",map_location='cpu').to(device)


def da_so(data):
    num=data.shape[0]
    d = data.reshape( [ num, 80, 6 ] )
    return d

def test(e):
    en = e * 60
    d0 = np.loadtxt('data/data/test.csv', delimiter=',')[en:en + 60]
    ad = da_so(d0)
    data = torch.FloatTensor(ad)
    return data


a1=0
ty=0

for lab in range( 8 ):
    data = test(lab).to(device)
    time_start = time.time()

    out = model(data)

 #   out = mix_model(data)

 #   data = torch.permute( data, [0,2,1] )
 #   _,out = model(data)
 #   print("Output dimensions:",out.size())

    y = out.argmax(1)
    yyy=list(y)
    acc = (out.argmax(1) == lab).sum()

    lt=[]
    for i in range(len(yyy)):
        if yyy[i] != lab :
            lt.append(i)

  #  print(y,acc,data.size(0))
    a1+=acc
    time_end = time.time()
    ty+=( time_end -time_start)
  #  print(lt)

print(a1/480)

print(ty)
