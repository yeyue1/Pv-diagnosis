import os
import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy.random import permutation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch import nn,Tensor
from torch.utils.data import TensorDataset
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce  # 张量操作

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
s=StandardScaler()

def s_100(data):
    num=data.shape[0]
    d = data.reshape([num, 80, 6])
    return d


def real_train():
    d0 = np.loadtxt('data/data/train.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/tr_la.csv', delimiter=',')
    label = label.reshape(-1)
    return d0,label


def real_val():
    d0 = np.loadtxt('data/data/val.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/va_la.csv', delimiter=',')
    label = label.reshape(-1)
    return d0,label


def real_test():
    d0 = np.loadtxt('data/data/test.csv', delimiter=',')
    d0 = s_100(d0)
    label = np.loadtxt('data/test_la.csv', delimiter=',')
    label = label.reshape(-1)
    return d0,label


def create_10s(x, digit_indices,num_classes):  # 构造对，交替生成正负样本对，输入x图像数据    及每类的索引标签
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []  # 存储样本对
    labels = []  # 存储样本对对应的标签
    n = min([len(digit_indices[d]) for d in range(num_classes)])  # 中括号里面计算每一类的样本数，n是最小的样本数-1
    for d in range(num_classes):  # 对每一类进行操作
        for i in range(n):
            inc = random.randrange(1, 3)      #随机数
            i1 = (i + inc) % n
            i2 = (i + 2*inc) % n
            i3 = (i + 3*inc) % n
            i4 = (i + 4 * inc) % n

            z11, z12 = digit_indices[d][i], digit_indices[d][i1]  #这是寻找同类图片的索引
            z21, z22 = digit_indices[d][i], digit_indices[d][i2]
            z31, z32 = digit_indices[d][i], digit_indices[d][i3]
            z41, z42 = digit_indices[d][i], digit_indices[d][i4]

            pairs += [[x[z11], x[z12]]]  # 获得每张图的标签后，用索引找到位置，然后和同类构成正样本对标签为1
            pairs += [[x[z21], x[z22]]]
            pairs += [[x[z31], x[z32]]]
            pairs += [[x[z41], x[z42]]]

            dn1 = (d + 1) % num_classes          #得到除法的余数
            dn2 = (d + 2) % num_classes
            dn3 = (d + 3) % num_classes
            dn4 = (d + 4) % num_classes

            a11, a12 = digit_indices[d][i], digit_indices[dn1][i]  #寻找不同类
            a21, a22 = digit_indices[d][i], digit_indices[dn2][i]
            a31, a32 = digit_indices[d][i], digit_indices[dn3][i]
            a41, a42 = digit_indices[d][i], digit_indices[dn4][i]

            pairs += [[x[a11], x[a12]]]  # 和不同的类构成负样本对标签为0
            pairs += [[x[a21], x[a22]]]
            pairs += [[x[a31], x[a32]]]
            pairs += [[x[a41], x[a42]]]

            labels += [0.,0.,0.,0.
                      ,1.,1.,1.,1.]   # 欧式距离，真0假1 ；曼哈顿 真1假0
    return np.array(pairs), np.array(labels)  # 返回样本对和标签


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


class S_Net(nn.Module):
    def __init__(self,in_channels, **kwargs):
        super(S_Net, self).__init__()
        self.pe= nn.GRU( input_size=6, hidden_size= 128, batch_first=True )
        self.cla=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear( 128, 32 ),
            nn.Linear(32, 8),
        )

    def forward_once(self, x):
        output,_ = self.pe(x)
        y = self.cla(output[:,-1,:])
        return y

    def forward(self, input1, input2 ):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        euclidean_distance = F.pairwise_distance( output1, output2, keepdim=True)

        return  torch.abs(euclidean_distance)


def ContrastiveLoss( euclidean_distance, label, mergin ):
    # 若距离大于阈值，则torch.clamp 输出0，表示该距离符合要求, 该函数类似取最大值
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp( mergin - euclidean_distance, min=0.0), 2)
                                  )
    return loss_contrastive*10



num_heads=5
depth=3
embed_size=10
patch_size=2
drop=0


class MultiHeadAttention(nn.Module):  # 与公式一样
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)  # n 是长度
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # 与公式同样，Q * K.T  ，是将多头分开后的乘， qd * kd 不是点乘，是阵乘
        # einsum 的第一个参数是字符串，"ik,kj-> ij " ，表示queries的 i 行 k 列与 keys 的 k 列 j 行对应元素相乘再相加
        # 输出作为结果的第 i 行 j 列元素，维度的索引号只能是26个英文字母 'a' - 'z' 之一

        scaling = self.emb_size**0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 是加入的特征提取层

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=embed_size, num_heads=num_heads, drop_p=0., forward_expansion=4, forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=3, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

    def forward(self, x):
        out = self.clshead(x)
        return out


class PatchEmbedding_Linear(nn.Module):
    # what are the proper parameters set here?
    def __init__(self, in_channels, patch_size, emb_size, seq_length):
        # self.patch_size = patch_size
        super().__init__()
        # change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            # (w s2) 是时间步长 ，(h s1)是高为1的图像，将步长切分为patch_size长的片段，并编码
            # 得到 批次*片段数量*片段与特征的结合
            nn.Linear(patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x
