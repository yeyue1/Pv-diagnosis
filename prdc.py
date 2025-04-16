"""
@article{ferjad2020icml,
  title = {Reliable Fidelity and Diversity Metrics for Generative Models},
  author = {Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  year = {2020},
  booktitle = {International Conference on Machine Learning},
}"""


import numpy as np
import torch
from numpy import empty
from scipy.linalg import sqrtm
from torch import nn
from scipy.spatial.distance import cdist
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


__all__ = ['compute_prdc']


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = cdist( data_x, data_y )
    return dists

def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values

def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    """
    real_features = real_features.cpu().detach().numpy()
    fake_features = fake_features.cpu().detach().numpy()
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances( real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances( fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance( real_features, fake_features )

   # precision = ( distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)  ).any(axis=0).mean()

  #  recall = ( distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0) ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * ( distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                                        ).sum(axis=0).mean()

    coverage = ( distance_real_fake.min(axis=1) < real_nearest_neighbour_distances ).mean()

    return density, coverage


def Kid( real_features, gen_features, num_subsets, max_subset_size):  # inc
    real_features = real_features.cpu().detach().numpy()
    gen_features = gen_features.cpu().detach().numpy()

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

def calculate_fid(act1, act2):  # inc
    act1 = act1.cpu().detach().numpy()
    act2 = act2.cpu().detach().numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




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
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
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

