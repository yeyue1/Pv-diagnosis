# coding:utf-8
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import misc
from train_data import  train_r
from model_loss import toggle_grad, four_three, gp_dis, gp_gen, cal_gp
from model import Generator, Discriminator, G_D
from APA_stats import Collector,adaptive_pseudo_augmentation,report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def few_sample( lab, d_n, apa=False ):
    generator = Generator( )
    discriminator = Discriminator()

    apa_stats = Collector(regex='Loss/signs/real')

    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    GAN = G_D(generator, discriminator)

    if lab==0:
        lrr = 0.000135
    elif lab ==3:
        lrr = 0.00014
    else:
        lrr = 0.00012

    beta = (0., 0.999)
    optimizer_G = torch.optim.Adam(generator.parameters(), weight_decay=1e-3, lr=lrr, betas=beta)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), weight_decay=1e-3, lr=0.0002, betas=beta)

    all_train = train_r(lab)
    dataloader = DataLoader(dataset=all_train, batch_size=6, shuffle=True, drop_last=False)

    total_epochs = 6000
    max_g=10000.0
    for epoch in range(total_epochs):  # 循环epoch
        D_loss_total = 0
        G_loss_total = 0

        for i, real_data in enumerate(dataloader):  # 循环iter
            real_data = real_data.to(device)
            bs = real_data.shape[0]
            z = torch.randn((bs, 100), dtype=torch.float).to(device)  # 生成输入噪声z，服从标准正态分布，长度为latent_dim

            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            fake_data = generator(z)
            f_data = four_three(fake_data)
            rd = deepcopy(real_data)

            if epoch % 3 == 0 and apa:
                real_data = adaptive_pseudo_augmentation(real_data, f_data, device, GAN.p)
                print(GAN.p)

            r_data = torch.permute(real_data,[0,2,1]).unsqueeze(2)
            rdd =  torch.permute(rd,[0,2,1]).unsqueeze(2)

            optimizer_D.zero_grad()  # 判别网络D清零梯度
            D_fake, D_real = GAN(fake_data, r_data)
            report('Loss/signs/real', D_real.sign())

            D_loss_real, D_loss_fake = gp_dis(D_real,D_fake,device)
            loss_D = D_loss_real + D_loss_fake + cal_gp( D=discriminator, real=rdd, fake=fake_data, cuda=cuda ) * 10
            loss_D.backward()
            D_loss_total += loss_D.item()

            if epoch % 3 == 0 and apa:
                apa_stats.update()  # 借助 loss.py 的 report 完成更新 ；np.sign 只有 1 和 -1
                adjust = np.sign( apa_stats['Loss/signs/real'] - 0.6 ) / 100
                GAN.p.copy_((GAN.p + adjust).max(misc.constant(0, device=device)))

            optimizer_D.step()

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)

            optimizer_G.zero_grad()  # 生成网络G清零梯度
            gz = generator( torch.randn( ( bs, 100), dtype=torch.float).to(device) )
            G_fake = GAN( gz )            # 噪声z输入生成网络G，得到判别器结果
            loss_G = gp_gen( G_fake, device )  # 生成网络G的损失函数
            loss_G.backward()  # 反向传播，计算当前梯度
            G_loss_total += loss_G.item()
            optimizer_G.step()  # 根据梯度，更新网络参数

        print("epoch:%.0f,total_D_loss:%.4f,total_G_loss:%.4f" % (
            epoch, D_loss_total / len(dataloader), G_loss_total / len(dataloader)))  # 显示当前epoch训练完成后，判别网络D和生成网络G的总损失

        if epoch>=total_epochs-1000 and G_loss_total / len(dataloader)<=max_g:
            max_g=G_loss_total / len(dataloader)
            torch.save(generator, "TTS_model/" + str(d_n) + "/APA_" + str(lab) + ".pth")
            epoch_dict[lab]=str(epoch)+'    ' + str(max_g)

if __name__ == "__main__":
    all_dict = {}
    for d in range( 3 ):
        epoch_dict = {}
        for lab in range( 8 ):
            few_sample(lab, d, apa=True)
            time.sleep(20)
        all_dict[d] = epoch_dict
        print(all_dict)






