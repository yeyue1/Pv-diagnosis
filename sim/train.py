# coding:utf-8
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import TensorDataset, DataLoader
from data_make import create_10s, ContrastiveLoss, real_train, real_val, S_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tran_tensor(data,lab):
    label = torch.FloatTensor(lab)
    dat = torch.FloatTensor(data)
    all = TensorDataset(dat, label)
    return all

def dis_trans(dis):
    batch_size = dis.size(0)
    alpha = torch.ones([batch_size], device=device)
    k = torch.where(dis >= 0.5, alpha, torch.zeros_like(alpha))
    return k


num_classes=8
org_data, org_label = real_train()
val_data, val_label = real_val()

train_indices = [np.where(org_label == i)[0] for i in range(num_classes)]   #np.where(条件)，返回满足条件的‘索引’，如arr = np.array([11, 3, 4, 5, 6, 7, 8, 9]) print(np.where(arr < 5))-(array([1, 2], dtype=int64),)
val_indices = [np.where(val_label == i)[0] for i in range(num_classes)]

train_pair, train_sl = create_10s(org_data, train_indices,num_classes)  # 相似为0 ，不同为1
val_pair, val_sl = create_10s(val_data, val_indices,num_classes)

data_tra=tran_tensor(train_pair,train_sl)
data_val=tran_tensor(val_pair,val_sl)


net = S_Net(in_channels=6).to(device)
optimizer = torch.optim.Adam(net.parameters(), 0.001 )

max_test = 0.5

for epoch in range(0, 1000):
    net.train()
    l = 0
    ac = 0
    train_load = DataLoader(dataset=data_tra, batch_size=64, shuffle=True, drop_last=False)
    for i, da_bz in enumerate(train_load, 0):
        img, tr_label = da_bz
        img0, img1 = img[:,0],img[:,1]
        tr_label=tr_label.to(device)

        optimizer.zero_grad()
        dis_tra = net(img0.to(device), img1.to(device))
        dis_tra=dis_tra.squeeze(-1)

        loss_con_tra = ContrastiveLoss( dis_tra, tr_label, mergin= 1. )

        loss_con_tra.backward()
        optimizer.step()
        l += loss_con_tra.item()

        acc = (dis_trans(dis_tra) == tr_label).sum()  # (1)是指对每一行搜索最大
        ac = ac + acc

    print("训练次数:{}, " 
          "Loss:{}".format(epoch, l / len(train_load)))
    print("        训练集上的正确率：{}".format(ac / len(data_tra)))

    net.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        # 这部分用于测试不用于训练所以不计算梯度
        test_load = DataLoader(dataset=data_val, batch_size=32, shuffle=True, drop_last=False)
        for i, d_v in enumerate(test_load):
            val_01, val_label = d_v
            val_0, val_1=val_01[:,0],val_01[:,1]
            val_label=val_label.to(device)

            dis_val = net(val_0.to(device), val_1.to(device))
            dis_val=dis_val.squeeze(-1)

            loss_con_val = ContrastiveLoss( dis_val, val_label, mergin= 1. )

            total_test_loss = total_test_loss + loss_con_val.item()

            val_acc = (dis_trans(dis_val) == val_label).sum()  # (1)是指对每一行搜索最大
            total_accuracy = total_accuracy + val_acc

    print("     测试集上的Loss:{}".format(total_test_loss / len(test_load)))
    print("     测试集上的正确率：{}".format(total_accuracy / len(data_val)))

    if total_test_loss / len(test_load) <= max_test and ac / len(data_tra) >= 0.99 :
        max_test = total_test_loss / len(test_load)
        torch.save(net, 'sim.pth')
        print('********'
            'DONE'
              '********')
