# coding:utf-8
import math
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce  # 张量操作

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

num_heads= 5
depth=3
embed_size= 10
patch_size=2
drop=0

class gru(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pe= nn.GRU( input_size=6, hidden_size= 128, batch_first=True )
        self.cla=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear( 128, 32 ),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        output,_ = self.pe(x)
        output = output[:, -1, :]
        y = self.cla(output)
        return y


class Generator(nn.Module):
    def __init__(self, seq_len=80, patch_size=patch_size, channels=6 , latent_dim=100,
                 embed_size=embed_size, depth=depth, num_heads=num_heads, forward_drop_rate=drop, attn_drop_rate=drop):
        super(Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim   # 随机噪声长度
        self.seq_len = seq_len
        self.embed_dim = embed_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(  depth=self.depth,  emb_size=self.embed_dim,  num_heads=self.num_heads,
                                               drop_p=self.attn_drop_rate,  forward_drop_p=self.forward_drop_rate)

        self.deconv = nn.Sequential( nn.Conv2d( self.embed_dim, self.channels, 1, 1, 0 ),
                                     nn.Tanh())

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, 1, self.seq_len)
        return output


class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=3, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])


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
                nn.Dropout(drop_p)            )            ))


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p))
            ),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p))            )        )


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
    def __init__(self, in_channels=21, patch_size=10, emb_size=100, seq_length=1024):
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


class Discriminator(nn.Sequential):  # patch_size 是 被切分 的步长的长度
    def __init__(self, in_channels=6, patch_size=patch_size, emb_size=embed_size, seq_length=80, depth=depth, drop=drop, n_classes=1, **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=drop, forward_drop_p=drop, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten




class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D
        self.p= torch.as_tensor(0.0)

    def forward(self, G_z, x=None):
        if x is not None:
            D_input = torch.cat([img for img in [G_z, x]], 0)  # 将 生成数据 和 真正数据 结合
            D_out = self.D(D_input)
            return torch.split(D_out, [G_z.shape[0], x.shape[0]])
        else:
            D_out = self.D(G_z)
            return D_out   #  输出的东西