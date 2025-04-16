"""bibtex
@inproceedings{jiang2021DeceiveD,
  title={{Deceive D: Adaptive Pseudo Augmentation} for {GAN} Training with Limited Data},
  author={Jiang, Liming and Dai, Bo and Wu, Wayne and Loy, Chen Change},
  booktitle={NeurIPS},
  year={2021}
}
"""

import re
import numpy as np
import torch
import dnnlib
import misc

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

@misc.profiled_function
def report(name, value):
    r"""跨设备和进程边界向“收集器”的所有感兴趣的实例广播给定的标量集。
该函数预计非常便宜，可以在训练循环、损失函数或`torch.nn.模块`。
警告：当前实现要求一组唯一名称在进程之间保持一致。请确保每个进程至少为每个唯一名称调用一次“report（）”，
调用顺序相同。如果给定的进程没有可广播的标量，它可以执行“report（name，[]）”（空列表）。
    Args:
        name：指定统计信息名称的任意字符串。每个唯一名称的平均值分别累加。
        value：任意一组标量。可以是列表、元组，NumPy数组、PyTorch张量或Python标量。
    Returns:
        The same `value` that was passed in.
    """
    if name not in _counters:
        _counters[name] = dict()

    elems = torch.as_tensor(value)   # 转换为 tensor
    if elems.numel() == 0:  # 返回数组中元素的个数
        return value

    elems = elems.detach().flatten().to(_reduce_dtype)
    moments = torch.stack([ torch.ones_like(elems).sum(),  elems.sum(),  elems.square().sum()  ])
    assert moments.ndim == 1 and moments.shape[0] == _num_moments
    moments = moments.to(_counter_dtype)

    device = moments.device
    if device not in _counters[name]:
        _counters[name][device] = torch.zeros_like(moments)
    _counters[name][device].add_(moments)   # 是 moments 里面的3个值，--字典
    return value

class Collector:
    r"""收集“report（）”和“report0（）”广播的标量，并计算它们在用户定义的时间段内的长期平均值（平均值和标准差）。
平均值首先被收集到用户不直接可见的内部计数器中。然后，由于调用“update（）”，
它们被复制到用户可见状态，然后可以使用“mean（）”、“std（））”、‘as_dict（）’等进行查询。
    Args:
        regex:          定义要收集的统计信息的正则表达式。默认情况是收集所有信息.
        keep_previous:  如果在给定的一轮中没有收集标量，是否保留以前的平均值      (default: True).
    """
    def __init__(self, regex='.*', keep_previous=True):
        self._regex = re.compile(regex)   # 收集所有信息
        self._keep_previous = keep_previous
        self._cumulative = dict()
        self._moments = dict()
        self.update()
        self._moments.clear()

    def names(self):
        r"""    返回到目前为止广播的与构造时指定的正则表达式匹配的所有统计信息的名称。    """
        return [name for name in _counters if self._regex.fullmatch(name)]

    def update(self):
        r"""将内部计数器的当前值复制到用户可见状态，并为下一轮重置它们。
如果在构造时指定了“keep_previous=True”，则对于自上次更新以来未接收到标量的统计信息，将跳过该操作，保留其以前的平均值。
此方法执行多个GPU到CPU的传输和一个“torch.distributed.all_reduce（）”。它打算在主训练循环中定期调用，通常每N个训练步骤调用一次
        """
        if not self._keep_previous:
            self._moments.clear()
        for name, cumulative in _sync(self.names()):
            if name not in self._cumulative:
                self._cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
                # torch.zeros 返回一个形状为为 _num_moments=3 , 里面的每一个值都是0的tensor
            delta = cumulative - self._cumulative[name]
            self._cumulative[name].copy_(cumulative)
            if float(delta[0]) != 0:
                self._moments[name] = delta

    def _get_delta(self, name):
        r"""    返回为给定最后两次调用“update（）”之间的统计信息，如果没有收集标量。     """
        assert self._regex.fullmatch(name)
        if name not in self._moments:
            self._moments[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        return self._moments[name]

    def num(self, name):
        r""" 返回在最后两次调用“update（）”之间为给定统计信息累积的标量数，如果未收集标量，则返回零。 """
        delta = self._get_delta(name)
        return int(delta[0])

    def mean(self, name):
        r""" 返回在最后两次调用“update（）”之间为给定统计信息累积的标量的平均值，如果没有收集标量，则返回NaN。 """
        delta = self._get_delta(name)
        if int(delta[0]) == 0:
            return float('nan')
        return float(delta[1] / delta[0])

    def std(self, name):
        r"""  返回在最后两次调用“update（）”之间为给定统计信息累积的标量的标准偏差，如果没有收集标量，则返回NaN。  """
        delta = self._get_delta(name)
        if int(delta[0]) == 0 or not np.isfinite(float(delta[1])):
            return float('nan')
        if int(delta[0]) == 1:
            return float(0)
        mean = float(delta[1] / delta[0])
        raw_var = float(delta[2] / delta[0])
        return np.sqrt(max(raw_var - np.square(mean), 0))

    def as_dict(self):
        r"""以`dnnlib.EasyDict`的形式返回最后两次调用update（）`之间累积的平均值。内容如下：
            dnnlib.EasyDict( NAME = dnnlib.EasyDict(num=FLOAT, mean=FLOAT, std=FLOAT),  ...   )        """
        stats = dnnlib.EasyDict()
        for name in self.names():
            stats[name] = dnnlib.EasyDict(num=self.num(name), mean=self.mean(name), std=self.std(name))
        return stats

    def __getitem__(self, name):
        r""" 方便吸收    `collector[name]`是collector.mean（name）`的同义词。 """
        return self.mean(name)


def _sync(names):
    r""" 跨设备和进程同步全局累积计数器。由`Collector.update（）`内部调用。 """
    if len(names) == 0:
        return []
    global _sync_called
    _sync_called = True

    deltas = []    #  收集当前rank内的增量。
    device = _sync_device if _sync_device is not None else torch.device('cpu')
    for name in names:
        delta = torch.zeros([_num_moments], dtype=_counter_dtype, device=device)
        for counter in _counters[name].values():
            delta.add_(counter.to(device))
            counter.copy_(torch.zeros_like(counter))
        deltas.append(delta)
    deltas = torch.stack(deltas)

    # Sum deltas across ranks.
    if _sync_device is not None:
        torch.distributed.all_reduce(deltas)

    # Update cumulative values.
    deltas = deltas.cpu()
    for idx, name in enumerate(names):
        if name not in _cumulative:
            _cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        _cumulative[name].add_(deltas[idx])

    # Return name-value pairs.
    return [(name, _cumulative[name]) for name in names]



def adaptive_pseudo_augmentation(real_data, fake_data, device, p):        # 应用自适应伪增强（APA）
    batch_size = real_data.shape[0]
    alpha = torch.ones([batch_size, 1, 1], device=device)
    alpha = torch.where(torch.rand([batch_size, 1, 1], device=device) < p, alpha, torch.zeros_like(alpha))
    # 当condition为真，返回 pseudo_flag 的值，否则返回 zeros 的值 , pseudo_flag 是多维Tensor
    if torch.allclose(alpha, torch.zeros_like(alpha)):  # 比较两个元素是否接近 , 当p很小时，alpha接近0
        return real_data
    else:
        return fake_data * alpha + real_data * (1 - alpha)