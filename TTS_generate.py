# coding:utf-8
import numpy as np
import pandas as pd
import torch
from prdc import cal_mmd, calculate_mmd
from train_data import train_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = "guan_"

for lab in range(8):
    model = torch.load("TTS_model/0/" + name + str(lab) + ".pth", map_location='cpu').to(device)

    z = torch.FloatTensor(np.random.normal(0, 1, (500, 100))).to(device)
    gen = model(z)
    gen = gen.squeeze(2).to(device)
    gen = torch.permute(gen, [0, 2, 1])
    org = train_r(lab).to(device)

    ml = []
    lis = []

    for ak in range(gen.size(0)):
        gu = gen[ak].unsqueeze(0).to(device)
        md = cal_mmd(gu, org, device)
        ml.append(md)

    sorted_id = sorted(range(len(ml)), key=lambda k: ml[k], reverse=True)
    s_id = sorted_id[:88]
    for ki in s_id:
        lis.append(gen[ki].squeeze(0).cpu().detach().numpy())

    emp = np.empty(shape=[(len(lis)) * 80, 6])
    for le in range(len(lis)):
        leg = le * 80
        emp[leg: leg + 80] = lis[le]

    ep = emp.reshape([len(lis), -1])
    org = org.cpu().detach().numpy()
    epm = np.concatenate([org.reshape([org.shape[0], -1]), ep], axis=0)

    gp = pd.DataFrame(epm)
    gp.to_csv('TTS_data/' + str(lab) + '.csv', index=False, header=False)


