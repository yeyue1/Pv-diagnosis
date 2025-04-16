# coding:utf-8
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model_loss import toggle_grad, test_sim, calculate_mmd, sim_mmd_loss, four_three,cal_mmd
from train_data import train_r
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)


def few_sample( lab, d_n ):
    gen_net = Generator()
    dis_net = torch.load("sim.pth", weights_only=False).to(device)

    gen_net.to(device)
    gen_net.apply(weights_init)

    gen_net.train()
    dis_net.train()
    for _,pa in dis_net.named_parameters():
        pa.requires_grad=False


    if lab==0:
        lrr = 0.000135
    elif lab ==3:
        lrr = 0.00014
    else:
        lrr = 0.00012

    optimizer_G = torch.optim.Adam( gen_net.parameters() , lr= lrr )

    all_train = train_r(lab)  # 12
    org_mmd = calculate_mmd(all_train, all_train, device)

    dataloader = DataLoader(dataset=all_train, batch_size= 6, shuffle=True, drop_last=False)
    total_epochs = 6000
    max_g=10000

    for epoch in range(total_epochs):  # 循环epoch
        G_loss_total = 0

        for i, real_data in enumerate(dataloader):  # 循环iter
            real_data = real_data.to(device)
            z = torch.FloatTensor(np.random.normal(0, 1, ( 6, 100 ) ) ).to(device)  # 生成输入噪声z，服从标准正态分布，长度为latent_dim

            fake_data = gen_net(z)
            # 输入序列视为高度为 1 的图像，输入的 时间步长 是图像宽度
            fake_data = four_three( fake_data )  # 转为 3 维的时间序列

            sim_dis = test_sim( real_data, fake_data, dis_net, device )

            mmd_dis_1 = cal_mmd( real_data, fake_data, device )  # 真实与虚假的分布差异
            mmd_dis_2 = calculate_mmd( fake_data ,fake_data, device )  # 虚假自身的分布差异

            loss_G =  sim_mmd_loss( sim_dis, mmd_dis_1, mmd_dis_2, org_mmd )
            print(sim_dis.item(), mmd_dis_1.item(), mmd_dis_2.item(),loss_G.item())

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            G_loss_total += loss_G.item()

        print("epoch:%.0f" % ( epoch))

        if epoch> total_epochs-1000 and G_loss_total / len(dataloader)<=max_g:
            max_g=G_loss_total / len(dataloader)
            torch.save( gen_net, 'TTS_model/' + str(d_n) + "/guan_" + str(lab) + ".pth")
            epoch_dict[lab]=str(epoch)+'    ' + str(max_g)

if __name__ == "__main__":
    all_dict = {}
    for d in range( 3 ):
        epoch_dict = {}
        for lab in range( 8 ):
            few_sample(lab, d)
            time.sleep(30)
        all_dict[d]=epoch_dict
        print(all_dict)