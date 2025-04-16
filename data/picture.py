# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ��������
n_d = np.loadtxt('data/noise/0.csv', delimiter=',')
d_n_d = np.loadtxt('data/de_noise/0.csv', delimiter=',')
nor_d = np.loadtxt('data/nor_data/0.csv', delimiter=',')

# ������Ƭ
n1 = n_d[800:880]
d1 = d_n_d[800:880]
nor1 = nor_d[800:880]

# �źű���
titles = [
    r'$G$',  # ʹ��LaTeX�﷨
    r'$T_a$',
    r'$U_oc$',
    r'$I_sc$',
    r'$U_m$',
    r'$I_m$'
]

# ����ͼ�δ�С����ͼ
fig, axs = plt.subplots(6, 1, figsize=(12, 18), dpi=80)

# ����ÿ����ͼ
for i in range(6):
    yl = range(1, 81)
    axs[i].plot(yl, nor1[:, i], label='normalized')  # ������ָ��label
   # axs[i].plot(yl, d1[:, i], label='delete_noise')  # ������ָ��label
    axs[i].set_title(titles[i], fontsize=15,fontfamily='Times New Roman')  # ���ñ���
    axs[i].tick_params(axis='x', which='both', bottom=False, labelbottom=(i == 5))  # ���һ����ʾx���ǩ
    axs[i].tick_params(axis='y', labelsize=12)
    axs[i].legend().set_visible(False)  # ����ÿ����ͼ��ͼ��

# ����ÿ����ͼ֮��ļ��
plt.subplots_adjust(hspace=0.4)  # ���� hspace ��ֵ�����Ӽ�ࣨԭ���� 1.0����Ϊ 2.0��

# ���һ���ܵ�ͼ����ֻ���������ͼ�����Ͻ�
#fig.legend(['noise', 'delete_noise'], loc='upper right', fontsize=12, bbox_to_anchor=(1, 1))
fig.legend(['normalized'], loc='upper right', fontsize=12, bbox_to_anchor=(1, 1))
# ��ʾͼ��
plt.show()


"""
plt.figure(figsize=(12,15), dpi=80)
plt.figure(1)
ax1 = plt.subplot(611)
yl = range(1, 81)
plt.plot(yl, nor1[:,0],c='r', label='nor')
plt.xticks([])
plt.yticks(fontproperties='Times New Roman', size=15)

ax2 = plt.subplot(612)
yl = range(1, 81)
plt.plot(yl, nor1[:,1],c='m', label='nor')
plt.xticks([])
plt.yticks(fontproperties='Times New Roman', size=15)

ax3 = plt.subplot(613)
yl = range(1, 81)
plt.plot(yl, nor1[:,2],c='g', label='nor')
plt.xticks([])
plt.yticks(fontproperties='Times New Roman', size=15)

ax4 = plt.subplot(614)
yl = range(1, 81)
plt.plot(yl, nor1[:,3],c='y', label='nor')
plt.xticks([])
plt.yticks(fontproperties='Times New Roman', size=15)

ax5 = plt.subplot(615)
yl = range(1, 81)
plt.plot(yl, nor1[:,4],c='blue', label='nor')
plt.xticks([])
plt.yticks(fontproperties='Times New Roman', size=15)

ax6 = plt.subplot(616)
yl = range(1, 81)
plt.plot(yl, nor1[:,5],c='b', label='nor')
plt.xticks(fontproperties='Times New Roman', size=15)
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xlabel('time spot', fontsize=15)

plt.show()
"""

