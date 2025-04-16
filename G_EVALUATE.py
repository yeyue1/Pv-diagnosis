# coding:utf-8
import numpy as np
import pandas as pd
import torch
from prdc import calculate_mmd, compute_prdc, Kid, calculate_fid
from train_data import train_r, test_sim, pre_allin,train_many

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solar = torch.load('data_mat/m6.pth' ,map_location='cpu').to(device)
sim_d=torch.load("sim.pth" , map_location='cpu' ).to(device)

sv = torch.load('data_mat/s_vgg.pth' , map_location='cpu' ).to(device)
si = torch.load('data_mat/s_inc.pth' , map_location='cpu' ).to(device)

fv = torch.load('data_mat/f_vgg.pth' , map_location='cpu' ).to(device)
fi = torch.load('data_mat/f_inc.pth' , map_location='cpu' ).to(device)

def SYD(f_p,f_m):
    if f_p<0.75:
        return 0
    else:
        return f_p+f_m

cho= 4

if cho==0:
    name = "/few_"   #   org = train_r(lab).to(device)

if cho==1:
    name = "/many_"   # for d_n in range( 1 );  org = train_many(lab).to(device)

elif cho==2:
    name = "/APA_"  #  org = train_r(lab).to(device)

elif cho==3:
    name = "/LC_"  #   org = train_r(lab).to(device)

elif cho==4:
    name = "/guan_"  #   org = train_r(lab).to(device)




for d_n in range( 3 ):
    for lab in range( 8 ):
        model = torch.load("TTS_model/" + str(d_n) + name + str(lab) + ".pth" ,map_location='cpu').to(device)

        z = torch.FloatTensor(np.random.normal(0, 1, (100, 100))).to(device)
        gen = model(z)
        gen = gen.squeeze(2).to(device)
        org = train_r(lab).to(device)
        org = torch.permute(org, [0, 2, 1])

        acc = 0
        for ak in range(gen.size(0)):
            gu = gen[ak].unsqueeze(0).to(device)
            gu =torch.permute(gu, [0, 2, 1])
            out = solar(gu)
            if out.argmax(1) == lab:
                acc += 1
  #      print('f_P:'+ str(acc))

        ge= torch.permute(gen, [0, 2, 1])
        ork = torch.permute(org, [0, 2, 1])
        m = calculate_mmd(ge, ge, device=device)
        o = calculate_mmd(ork, ork, device=device)
        fr_mmd = ( float(m)-float(o) )/float(o)
  #      print('fr_mmd:'+str(fr_mmd))
  #      print(m,o)

        test_data = torch.permute(gen, [0, 2, 1]).cpu().detach().numpy()
        s0 = train_r(0).to(device).cpu().detach().numpy()
        s1 = train_r(1).to(device).cpu().detach().numpy()
        s2 = train_r(2).to(device).cpu().detach().numpy()
        s3 = train_r(3).to(device).cpu().detach().numpy()
        s4 = train_r(4).to(device).cpu().detach().numpy()
        s5 = train_r(5).to(device).cpu().detach().numpy()
        s6 = train_r(6).to(device).cpu().detach().numpy()
        s7 = train_r(7).to(device).cpu().detach().numpy()
        p0 = test_sim(s0, test_data, sim_d, device)
        p1 = test_sim(s1, test_data, sim_d, device)
        p2 = test_sim(s2, test_data, sim_d, device)
        p3 = test_sim(s3, test_data, sim_d, device)
        p4 = test_sim(s4, test_data, sim_d, device)
        p5 = test_sim(s5, test_data, sim_d, device)
        p6 = test_sim(s6, test_data, sim_d, device)
        p7 = test_sim(s7, test_data, sim_d, device)

        index = pre_allin(p0, p1, p2, p3, p4, p5, p6, p7)

        sa = 0
        i = 0
        for i in range(index.shape[0]):
            if index[i] == lab:
                sa += 1
        sak = sa / 100

        syd=SYD(sak,fr_mmd)

    #    print('sim_d:'+ str(sak))
    #    print('syd:'+ str(syd))


        re_s_v,_ = sv(org)
        fa_s_v,_ = sv(gen)
        re_s_i,_ = si(org)
        fa_s_i,_ = si(gen)
        s_d,s_c = compute_prdc(re_s_v,  fa_s_v, nearest_k=5)
        s_k = Kid( re_s_i, fa_s_i,num_subsets=2,max_subset_size=6 )
        s_f = calculate_fid( re_s_i, fa_s_i)
        
        
    #    print('      '            'source:')
    #    print('s_f:' + str(s_f))
    #    print('s_k:' + str(s_k))
    #    print('s_d:' + str(s_d))
    #    print('s_c:' + str(s_c))

        re_f_v,_ = fv(org)
        fa_f_v,_ = fv(gen)
        re_f_i,_ = fi(org)
        fa_f_i,_ = fi(gen)
        f_d,f_c = compute_prdc(re_f_v,  fa_f_v, nearest_k=5)
        f_k = Kid( re_f_i, fa_f_i,num_subsets=2,max_subset_size=6 )
        f_f = calculate_fid( re_f_i, fa_f_i)
    #    print('     '            'full_d:')
    #    print('f_f:' + str(f_f))
    #    print('f_k:' + str(f_k))
    #    print('f_d:' + str(f_d))
    #    print('f_c:' + str(f_c))
    



        print( round( acc,2 ),
               round(fr_mmd, 2),
               round(sak, 2),
               round(syd, 2),

               round(s_f, 0),
               round(f_f, 0),
               round(s_k, 2),
               round(f_k, 2),
               round(s_d, 2),
               round(f_d, 2),
               round(s_c, 2),
               round(f_c, 2),
               )

    print('*********'
          '*********')




