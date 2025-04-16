# coding:utf-8
from math import exp
import torch.nn.functional as F
import torch
from torch import nn,empty
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off   # false 不需要更新网络参数

def four_three(data):
    real_i = data.squeeze(2)
    real_i = torch.permute(real_i, [0, 2, 1])
    return real_i


def cal_gp(D, real, fake, cuda):  # 定义函数，计算梯度惩罚项gp
    real_i = four_three(real)
    fake_i = four_three(fake)

    r = torch.rand(real_i.size(0), 1,1)  # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
    if cuda:  # 如果使用cuda
        r = r.cuda()  # r加载到GPU
    x = (r * real_i + (1 - r) * fake_i).requires_grad_(True)  # 输入样本x，由真假样本按照比例产生，需要计算梯度

    x1=torch.permute(x,[0,2,1]).unsqueeze(2)
    d = D(x1)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    if cuda:  # 如果使用cuda
        fake = fake.cuda()  # fake加载到GPU
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake   )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||2-1)^2 的均值
    return gp


# 将EMA应用于损失的简单包装。
class ema_losses(object):
    def __init__(self, init=10., decay=0.99):
        self.G_loss = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay

    def update(self, cur, mode, itr):
        if itr < 1:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)

em=ema_losses()

def loss_dis(dis_fake, dis_real, epoch, ema=em):   # 目前使用的
    ema.update(torch.mean(dis_fake).item(), 'D_fake', epoch)  # self.D_fake * 0.9 + torch.mean(dis_fake) *(1 - 0.9)
    ema.update(torch.mean(dis_real).item(), 'D_real', epoch)  # self.D_real * 0.9 + torch.mean(dis_real) *(1 - 0.9)

    loss_real = -torch.mean(dis_real)
    loss_fake = torch.mean(dis_fake)
    return loss_real,loss_fake

def loss_hinge_dis(dis_fake, dis_real, epoch, ema=em):
    ema.update(torch.mean(dis_fake).item(), 'D_fake', epoch)
    ema.update(torch.mean(dis_real).item(), 'D_real', epoch)

    loss_real = F.relu(1. - dis_real)
    loss_fake = F.relu(1. + dis_fake)
    return torch.mean(loss_real), torch.mean(loss_fake)


def loss_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

def lecam_reg(dis_real, dis_fake, ema=em):
    reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
    return reg

"""
@inproceedings{lecamgan,
  author = {Tseng, Hung-Yu and Jiang, Lu and Liu, Ce and Yang, Ming-Hsuan and Yang, Weilong},
  title = {Regularing Generative Adversarial Networks under Limited Data},
  booktitle = {CVPR},
  year = {2021}
}
"""

# Default to hinge loss
gen_loss = loss_gen
dis_loss = loss_dis
lc_loss  = loss_hinge_dis


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma, device):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total=total.to(device)
        total0 = total.unsqueeze(0).expand( int(total.size(0)), int(total.size(0)), int(total.size(1) ) )
        total0=total0.to(device)
        total1 = total.unsqueeze(1).expand( int(total.size(0)), int(total.size(0)), int(total.size(1) ) )
        total1=total1.to(device)
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp.to(device)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target , device):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma ,device=device)
        XX = torch.mean(kernels[:batch_size, :batch_size]).to(device)
        YY = torch.mean(kernels[batch_size:, batch_size:]).to(device)
        XY = torch.mean(kernels[:batch_size, batch_size:]).to(device)
        YX = torch.mean(kernels[batch_size:, :batch_size]).to(device)
        loss = torch.mean(XX + YY - XY - YX).to(device)
        return loss

def calculate_mmd( real_data, fake_data, device ):
    MMD = MMDLoss()
    kp = empty( [real_data.size(0), fake_data.size(0)] )
    real=torch.permute(real_data,[0,2,1])
    fake=torch.permute(fake_data,[0,2,1])

    for n, target in enumerate(real):
        for m, source in enumerate(fake):
            kp[n, m] = MMD(source=source, target=target, device=device)
    return kp.mean()


def cal_mmd( real_data, fake_data, device ):
    MMD = MMDLoss()
    real=real_data.reshape([ real_data.size(0) , -1 ])
    fake=fake_data.reshape([ fake_data.size(0) , -1 ])
    return MMD(source=real, target=fake, device=device)


def hinge_dis( real_validity, fake_validity ):
    d_loss= torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) +torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    return d_loss


def hinge_gen( fake_validity ):
    return -torch.mean(fake_validity)


def mse_dis( real_validity, fake_validity, device ):
    real_label = torch.full((real_validity.shape[0], real_validity.shape[1]), 1., dtype=torch.float,
                            device= device )
    fake_label = torch.full((fake_validity.shape[0], fake_validity.shape[1]), 0., dtype=torch.float,
                            device = device )
    d_real_loss = nn.MSELoss()(real_validity, real_label)
    d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
    return d_real_loss , d_fake_loss


def mse_gen( fake_validity, device ):
    real_label = torch.full(( fake_validity.shape[0], fake_validity.shape[1]), 1., dtype=torch.float, device= device )
    g_loss = nn.MSELoss()(fake_validity, real_label)
    return g_loss


def gp_dis( real, fake ,device):
    return -torch.mean( real ), torch.mean( fake )


def gp_gen(  fake, device ):
    return -torch.mean( fake  )


def test_sim(support,data,model ,device):  # 将每个支持数据和故障数据合成一对
    pairs = torch.empty([support.size(0),data.size(0)])  # 存储样本对
    for d in range(support.shape[0]):  # 样本集的每个样本
        for i in range(data.shape[0]):  #支持集的每个样本
            pre = model( support[d].unsqueeze(0).to( device ), data[i].unsqueeze(0).to( device ) )   #输出成1列
            pairs[d][i]=pre

    return pairs.max()  #行为与支持的相似度，列为数据样本


def im_tanh( x , k ):
    x = 10 * ( x - k )
    ans = exp(x) / ( exp(x) + 1 )
    return ans


def sim_mmd_loss( sim ,mmd_1, mmd_2 ,org_mmd ):  # 双曲函数tanh 和反比例函数 , k是重点
    org_m = float( org_mmd )
    sym = float(mmd_1) * 0.5 + float(sim)
    lamb = im_tanh( sym, k = 0.5 )

    beta =  1 - lamb
    m2_num =  org_m /  ( mmd_2 + 1e-10 )

    loss = sim + mmd_1 * 0.5 + beta * m2_num * org_m * 2
    return loss