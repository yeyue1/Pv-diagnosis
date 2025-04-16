# coding:utf-8
import math

import numpy as np
import pandas as pd
import pywt
from PyEMD import CEEMDAN
from scipy.stats import spearmanr


def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    data = list(new_df)  # 将np.ndarray()转为列表
    w = pywt.Wavelet('dB10')#选择dB10小波基
    ca3, cd3, cd2, cd1 = pywt.wavedec(data, w, level=3)  # 3层小波分解

    length1 = len(cd1)
    length0 = len(data)

    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))
    usecoeffs = []
    usecoeffs.append(ca3)

    #软阈值方法
    for k in range(length1):
        if (abs(cd1[k]) >= lamda/np.log2(2)):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda/np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda/np.log2(3)):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda/np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda/np.log2(4)):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda/np.log2(4))
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


def ceemd_deal( data ):
    ceemd = CEEMDAN()
    ceemd.ceemdan(data)
    imfs_ceemd, _ = ceemd.get_imfs_and_residue()

    al=[]
    for nk in range( imfs_ceemd.shape[0] ):
        r=spearmanr( data, imfs_ceemd[nk] ).statistic
        al.append(r)
    num = al.index(min(al))

    for i in range(len(al)):
        if i <= num:
            imfs_ceemd[i]=wavelet_noising(imfs_ceemd[i])
        else:
            continue

    CEEmd_out3 = np.zeros(80, )
    for n in range(len(al)):
        CEEmd_out3 += imfs_ceemd[n]
    return CEEmd_out3



for lab in range(8):
    da = np.loadtxt( 'data/noise/' + str(lab) + '.csv' ,delimiter=',' )
    ne = np.empty(shape=[da.shape[0], da.shape[1]])
    print(da.shape[0])
    for le in range( int( da.shape[0]//80 ) ):
        leg = le * 80
        data = da[ leg : leg + 80 ]

        for ni in range(data.shape[1]):
           ne[  leg : leg + 80 , ni ] = ceemd_deal( data[ : , ni ] )

    f_lis = pd.DataFrame(ne)
    f_lis.to_csv( 'de_noise/' + str(lab) + '.csv' , header=False, index=False )

