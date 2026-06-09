import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from mpm_solver_warp.mpm_data_structure import *

class Young_Moudulous_Map(torch.nn.Module):
    
    def __init__(self, E_item, inverted_index, gs_num):
        super(Young_Moudulous_Map, self).__init__()
        self.E = torch.nn.Parameter(E_item)
        self.inverted_index = inverted_index
        self.gs_num = gs_num

    def forward(self):
        E_out = torch.zeros(self.gs_num, dtype=torch.float)
        for i in range(0, self.gs_num):
            E_out[i] = self.E[self.inverted_index[i]]
        return E_out
    
class LinearStepAnneal(object):
    # def __init__(self, total_iters, start_state=[0.02, 0.98], end_state=[0.50, 0.98]):
    def __init__(
        self,
        total_iters,
        start_state=[0.02, 0.98],
        end_state=[0.02, 0.98],
        plateau_iters=-1,
        warmup_step=300,
    ):
        self.total_iters = total_iters

        if plateau_iters < 0:
            plateau_iters = int(total_iters * 0.2)

        if warmup_step <= 0:
            warmup_step = 0

        self.total_iters = max(total_iters - plateau_iters - warmup_step, 10)

        self.start_state = start_state
        self.end_state = end_state
        self.warmup_step = warmup_step

    def compute_state(self, cur_iter):

        if self.warmup_step > 0:
            cur_iter = max(0, cur_iter - self.warmup_step)
        if cur_iter >= self.total_iters:
            return self.end_state
        ret = []
        for s, e in zip(self.start_state, self.end_state):
            ret.append(s + (e - s) * cur_iter / self.total_iters)
        return ret

def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    From diffusers.optimization
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# below for compute batched SSIM
def gaussian(window_size, sigma):

    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def compute_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# above for compute batched SSIM