# coding:utf-8
import numpy as np
import pandas as pd
from mpmath import ln
from numpy.linalg import inv

"""
115.05  94.5  36.5  34.3 
103	115.6307335	3.693154836	90.07039233	2.634871772	15.52867593
371	117.9013222	13.37794047	91.68345045	11.69178729	18.21314768
495	118.6183154	17.8585237	91.50327	15.94264268	19.45245323

"""
def v(vv = 94.5 ):
    v1 = 90.07039233
    v2 = 91.68345045
    v3 = 91.50327

    g1 = 103
    g2 = 371
    g3 = 495

    t1 = 15.52867593
    t2 = 18.21314768
    t3 = 19.45245323

    W = np.array([[ln(g1 / 1000), (t1 - 25), (t1 - 25) * (g1 / 1000)],
                  [ln(g2 / 1000), (t2 - 25), (t2 - 25) * (g2 / 1000)],
                  [ln(g3 / 1000), (t3 - 25), (t3 - 25) * (g3 / 1000)]], dtype=float)

    result = np.array([ v1 -vv, v2 -vv, v3 -vv])

    W_inv = inv(W)
    vxyz = np.dot(W_inv, result)
    v = list(vxyz)
    print("data：")
    print(v[0])
    print(v[1])
    print(v[2])


"""
115.05  94.5  36.5  34.3 
103	115.6307335	3.693154836	90.07039233	2.634871772	15.52867593
371	117.9013222	13.37794047	91.68345045	11.69178729	18.21314768
495	118.6183154	17.8585237	91.50327	15.94264268	19.45245323

"""

def I(vv = 34.3  ):
    v1 = 2.634871772
    v2 = 11.69178729
    v3 = 15.94264268

    g1 = 103
    g2 = 371
    g3 = 495

    t1 = 15.52867593
    t2 = 18.21314768
    t3 = 19.45245323

    W = np.array([[ (g1 / 1000)*vv, (t1 - 25), (t1 - 25) * (g1 / 1000)],
                  [ (g2 / 1000)*vv, (t2 - 25), (t2 - 25) * (g2 / 1000)],
                  [ (g3 / 1000)*vv, (t3 - 25), (t3 - 25) * (g3 / 1000)]], dtype=float)

    result = np.array([ v1 , v2 , v3 ])

    W_inv = inv(W)
    vxyz = np.dot(W_inv, result)
    v = list(vxyz)
    print("data：")
    print(v[0])
    print(v[1])
    print(v[2])


def voc_norm(data ,vm=115.05 ,a1=-2.368468991272696 ,a2=0.7304617459063127 ,a3=-2.1686115343229444 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,1]-a1*float(ln(data[n,0]/1000))-a2*(data[n,5]-25)-a3*(data[n,5]-25)*(data[n,0]/1000)
        num=vn/vm
        list.append(2*num-1)
    return np.array(list)

def vm_norm(data ,vm=94.5 ,a1=6.205711413664325 ,a2=-1.225309548798287 ,a3=1.9775239166121565 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,3]-a1*float(ln(data[n,0]/1000))-a2*(data[n,5]-25)-a3*(data[n,5]-25)*(data[n,0]/1000)
        num = vn / vm
        list.append(2 * num - 1)
    return np.array(list)

def isc_norm(data ,im=36.5 ,b1= 0.9891563796383078 ,b2=0.0027917004726427663 ,b3=-0.0008841472337337564 ):
    list=[]
    for n in range(data.shape[0]):
        vn=data[n,2]-b2*(data[n,5]-25)-b3*(data[n,5]-25)*(data[n,0]/1000)
        vnt=vn*1000/(data[n,0]*b1)
        list.append(2*(vnt/im)-1)
    return np.array(list)

def im_norm(data ,im=34.3 ,b1=0.983992923075439 ,b2=0.07599899091781981 ,b3=0.1247154994887012 ):
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
    data = np.loadtxt('mat/'+str(lab)+'.csv', delimiter=',')
    irr=irr_norm(data).reshape([-1, 1])
    voc = voc_norm(data).reshape([-1, 1])
    isc = isc_norm(data).reshape([-1, 1])
    vm = vm_norm(data).reshape([-1, 1])
    im = im_norm(data).reshape([-1, 1])
    ff=ff_norm(data).reshape([-1, 1])

    fin = np.concatenate([irr, voc, isc, vm, im,ff], axis=1)
    fina=pd.DataFrame(fin)
    fina.to_csv('nor_data/'+str(lab)+'.csv', header=False,index=False)


