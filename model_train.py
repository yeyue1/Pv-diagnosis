# coding:utf-8
import numpy as np
import pandas as pd
from vgg_IV3 import VGG16,InceptionV3
import torch
from torch.utils.data import DataLoader
from model import  gru
from train_data import mat_train,real_train,real_val, real_many,hunhe_train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

net = 1
save_dir = ""
all_train = 0
all_test = 0
num_epoch = 0
train_batch=0

if __name__ == '__main__':

    choose = 2   # 0-few , 1-full , 2-sou_d , 3-sou_d, 5-mix_d

    if choose == 0:
        all_train = real_train()
        all_test = real_val()
        net= gru( in_channels=6 )
        save_dir="data_mat/tra.pth"
        num_epoch = 500
        train_batch=16

    elif choose == 1:
        all_train = real_many()
        all_test = real_val()
        net= gru( in_channels=6 )
        save_dir="data_mat/m6.pth"
        num_epoch = 500
        train_batch=64

    elif choose == 2:
        all_train = mat_train()
        all_test = real_val()
        net=VGG16()
        save_dir="data_mat/s_vgg.pth"
        num_epoch = 500
        train_batch=64

    elif choose == 3:
        all_train = mat_train()
        all_test = real_val()
        net=InceptionV3()
        save_dir="data_mat/s_inc.pth"
        num_epoch = 500
        train_batch=64

    elif choose == 8:
        all_train = real_many()
        all_test = real_val()
        net=VGG16()
        save_dir="data_mat/f_vgg.pth"
        num_epoch = 500
        train_batch=64

    elif choose == 9:
        all_train = real_many()
        all_test = real_val()
        net=InceptionV3()
        save_dir="data_mat/f_inc.pth"
        num_epoch = 500
        train_batch=64


    elif choose == 5:
        all_train = hunhe_train()
        all_test = real_val()
        net= gru( in_channels=6 )
        save_dir="data_mat/hunhe.pth"
        num_epoch = 500
        train_batch=64

    # 这是保存原始参数
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001 )
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    t_loss = []
    e_loss=[]

    max_test= 100.0
    # 训练网络
    for epoch in range(num_epoch):
        net.train()
        l = 0
        ac = 0
        train_load = DataLoader(dataset=all_train, batch_size=train_batch, shuffle=True, drop_last=False)
        for i, da in enumerate(train_load):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
            input_loader, label_loader = da

            if choose==2 or choose==3 or choose==9 or choose==8 :
                input_loader = torch.permute(input_loader,[0,2,1])

            input_loader, label_loader =input_loader.to(device), label_loader.to(device)
            out = net.forward(input_loader)
            if choose==2 or choose==3 or choose==9 or choose==8 :
                out = out[1]
            loss = loss_func(out, label_loader).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l+=loss.item()
            acc = (out.argmax(1) == label_loader).sum()  # (1)是指对每一行搜索最大
            ac =ac + acc
        print("训练次数:{}, "
              "Loss:{}".format(epoch, l/len(train_load)))
        t_loss.append(l/len(train_load))
        print("        训练集上的正确率：{}".format(ac / len(all_train)))

        net.eval()
        total_test_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            # 这部分用于测试不用于训练所以不计算梯度
            test_load = DataLoader(dataset=all_test, batch_size=16, shuffle=True, drop_last=False)
            for i, d_t in enumerate(test_load):
                imgs, targets = d_t

                if choose == 2 or choose == 3 or choose == 9 or choose == 8:
                    imgs = torch.permute(imgs, [0, 2, 1])

                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net.forward(imgs)
                if choose == 2 or choose == 3 or choose == 9 or choose == 8:
                    outputs = outputs[1]
                loss = loss_func(outputs, targets).to(device)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()  # (1)是指对每一行搜索最大
                total_accuracy = total_accuracy + accuracy

        print("     测试集上的Loss:{}".format(total_test_loss/len(test_load)))
        e_loss.append(total_test_loss/len(test_load))
        print("     测试集上的正确率：{}".format(total_accuracy / len(all_test)))
        #保存模型

        if total_test_loss/len(test_load)<max_test and ac/len(all_train)>0.9:
            max_test=total_test_loss/len(test_load)
            torch.save(net, save_dir)
            print('\n'+"{:*^30}".format(str(total_accuracy / len(all_test)))+'\n')

    if choose == 0 or choose == 1:
        t_l = np.array(t_loss).reshape([-1,1])
        e_l = np.array(e_loss).reshape([-1,1])
        et_l = np.concatenate([t_l,e_l],axis=1)
        etl = pd.DataFrame(et_l)
        etl.to_csv(str(choose)+'x.csv',header=False,index=False) # loss of  full_d and tra_d

    elif choose == 5:
        e_l = np.array(e_loss).reshape([-1,1])
        t_l = np.array(t_loss).reshape([-1,1])
        l_l = np.concatenate( [ e_l,t_l], axis=1)
        ll = pd.DataFrame(l_l)
        ll.to_csv('mix_loss.csv',header=False,index=False)   # loss of mix_d





