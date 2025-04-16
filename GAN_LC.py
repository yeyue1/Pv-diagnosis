# coding:utf-8
import time
import torch
from torch.utils.data import DataLoader
from train_data import train_r
from model_loss import gen_loss, toggle_grad, lecam_reg, lc_loss
from model import Generator, Discriminator, G_D
from APA_stats import report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

def few_sample( lab, d_n, apa=False ):
    generator = Generator( )
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()
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
            real_data = torch.permute(real_data,[0,2,1]).unsqueeze(2).to(device)
            bs = real_data.shape[0]
            z = torch.randn((bs, 100), dtype=torch.float).to(device)  # 生成输入噪声z，服从标准正态分布，长度为latent_dim

            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            fake_data = generator(z)

            optimizer_D.zero_grad()  # 判别网络D清零梯度
            D_fake, D_real = GAN(fake_data, real_data)
            report('Loss/signs/real', D_real.sign())
            D_loss_real, D_loss_fake = lc_loss(D_fake, D_real,epoch)
            reg=lecam_reg(D_real,D_fake)
            print(float(reg))

            loss_D = D_loss_real + D_loss_fake + reg * 0.3
            loss_D.backward()
            D_loss_total += loss_D.item()
            optimizer_D.step()

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)

            optimizer_G.zero_grad()  # 生成网络G清零梯度
            gz = generator( torch.randn((bs, 100), dtype=torch.float).to(device) )
            G_fake = GAN( gz )            # 噪声z输入生成网络G，得到判别器结果
            loss_G = gen_loss(G_fake)  # 生成网络G的损失函数
            loss_G.backward()  # 反向传播，计算当前梯度
            G_loss_total += loss_G.item()
            optimizer_G.step()  # 根据梯度，更新网络参数

        print("epoch:%.0f,total_D_loss:%.4f,total_G_loss:%.4f" % (
            epoch, D_loss_total / len(dataloader), G_loss_total / len(dataloader)))  # 显示当前epoch训练完成后，判别网络D和生成网络G的总损失

        if  epoch>=total_epochs-1000 and G_loss_total / len(dataloader)<=max_g:
            max_g=G_loss_total / len(dataloader)
            torch.save(generator, "TTS_model/" + str(d_n) + "/LC_" + str(lab) + ".pth")
            epoch_dict[lab]=str(epoch)+'    ' + str(max_g)

if __name__ == "__main__":
    all_dict = {}
    for d in range(2,3):
        epoch_dict = {}
        for lab in range(8 ):
            few_sample(lab, d, apa=False)
            time.sleep(20)
        all_dict[d] = epoch_dict
        print(all_dict)






