# coding:utf-8
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model_loss import toggle_grad, mse_gen, mse_dis,gp_gen,gp_dis,cal_gp
from train_data import train_r,train_many
from model import Generator, Discriminator, copy_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


num_heads=5
depth=3
embed_size=10
patch_size=2
drop=0


def few_sample( lab, d_n ):
    gen_net = Generator( patch_size=patch_size, embed_size=embed_size, depth=depth  , num_heads=num_heads)
    dis_net = Discriminator()

    gen_net.to(device)
    dis_net.to(device)
    gen_net.train()
    dis_net.train()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    if lab==0:
        lrr = 0.000135
    elif lab ==3:
        lrr = 0.00014
    else:
        lrr = 0.00012

    beta = (0., 0.999)
    optimizer_G = torch.optim.Adam( filter( lambda p: p.requires_grad, gen_net.parameters() ) ,
                                    lr= lrr, weight_decay=1e-3, betas=beta )
    optimizer_D = torch.optim.Adam( filter( lambda p: p.requires_grad, dis_net.parameters() ) ,
                                    lr= 0.0002, weight_decay=1e-3, betas=beta )

    all_train = train_r(lab)  # 150个
    dataloader = DataLoader(dataset=all_train, batch_size= 6, shuffle=True, drop_last=False)
    total_epochs = 6000
    max_g=10000
    for epoch in range(total_epochs):  # 循环epoch
        D_loss_total = 0
        G_loss_total = 0

        for i, real_data in enumerate(dataloader):  # 循环iter
            real_data =torch.permute(real_data,[0,2,1]).unsqueeze(2)
            r_data = real_data.to(device)
            bs = r_data.shape[0]
            z = torch.FloatTensor(np.random.normal(0, 1, (bs, 100))).to(device)  # 生成输入噪声z，服从标准正态分布，长度为latent_dim

            toggle_grad(gen_net, False)
            toggle_grad(dis_net, True)
            fake_data = gen_net(z)
            # 输入序列视为高度为 1 的图像，输入的 时间步长 是图像宽度

            optimizer_D.zero_grad()  # 判别网络D清零梯度
            D_fake = dis_net(fake_data)
            D_real = dis_net(r_data)

            D_loss_real, D_loss_fake = gp_dis(D_real,D_fake,device)
            loss_D = D_loss_real + D_loss_fake + cal_gp( D=dis_net, real=r_data, fake=fake_data, cuda=cuda ) * 10

            loss_D.backward()
            D_loss_total += loss_D.item()
            optimizer_D.step()

            toggle_grad(gen_net, True)
            toggle_grad(dis_net, False)

            optimizer_G.zero_grad()  # 生成网络G清零梯度
            gz = gen_net( torch.FloatTensor(np.random.normal(0, 1, ( bs, 100 ) ) ).to(device)  )

            loss_G = gp_gen( dis_net( gz ), device )  # 生成网络G的损失函数
            loss_G.backward()  # 反向传播，计算当前梯度
            G_loss_total += loss_G.item()
            optimizer_G.step()  # 根据梯度，更新网络参数

        print("epoch:%.0f,total_D_loss:%.4f,total_G_loss:%.4f" % (
            epoch, D_loss_total / len(dataloader), G_loss_total / len(dataloader)))  # 显示当前epoch训练完成后，判别网络D和生成网络G的总损失

        if epoch> total_epochs-1000 and G_loss_total / len(dataloader)<=max_g:
            max_g=G_loss_total / len(dataloader)
            torch.save( gen_net, 'TTS_model/' + str(d_n) + "/few_" + str(lab) + ".pth")
            epoch_dict[lab]=str(epoch)+'    ' + str(max_g)

if __name__ == "__main__":
    all_dict = {}
    for d in range(3):
        epoch_dict = {}
        for lab in range( 8 ):
            few_sample(lab, d)
            time.sleep(20)
        all_dict[d]=epoch_dict
        print(all_dict)