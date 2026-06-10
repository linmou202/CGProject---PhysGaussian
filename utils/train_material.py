import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import numpy as np
import argparse
import torch
import os
from time import time
import numpy as np

from typing import NamedTuple
import torch.nn.functional as F

from mpm_solver_warp.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)
from mpm_solver_warp.mpm_solver_diff import MPMWARPDiff
from mpm_solver_warp.warp_utils import from_torch_safe
from mpm_solver_warp.gaussian_sim_utils import get_volume
import warp as wp
import random

from utils.interface import (
    MPMDifferentiableSimulationClean,
    Calculate_Cov_and_Rot
)

from utils.render_utils import *
from utils.train_utils import *

class Trainer:
    def __init__(
            self,
            stage_renderer,
            init_cov,
            shs,
            opacity,
            mpm_state,
            mpm_model,
            mpm_solver,
            E_items,
            density_value,
            inverted_index,
            reference_path,
            frame_length = 0.02,
            num_frames = 50,
            ssim = 0.9,                        # the ratio between L1 loss and ssim loss
            parameter_scale = 1000,            # the scaling of E
            max_grad_norm = 1.0,
            gradient_accumulation_steps = 1,   # after how many backwards we step once
            warmup_step = 5,
            train_iters = 10,
            lr = 1e-3,
            device = "cuda:0"
        ):
        print("""initializing trainer..""")
        self.ssim = ssim
        self.parameter_scale = parameter_scale
        warmup_step = int(warmup_step * gradient_accumulation_steps)
        train_iters = int(train_iters * gradient_accumulation_steps)

        # setup the gaussians
        self.particle_init_position = mpm_solver.export_particle_x_to_torch(mpm_state, mpm_model).detach().clone()
        self.stage_renderer = stage_renderer
        self.reference_path = reference_path
        self.init_cov = init_cov
        self.opacity = opacity
        self.shs = shs

        self.num_frames = int(num_frames)
        self.frame_length = frame_length
        self.window_size_schduler = LinearStepAnneal(
            train_iters,
            start_state=[int(num_frames / 4)],
            end_state=[num_frames],
            plateau_iters=-1,
            warmup_step=3,
        )

        self.train_iters = train_iters
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # init traiable params
        E_module = Young_Moudulous_Map(E_items, inverted_index, self.particle_init_position.shape[0], device)
        self.E_module = E_module

        # setup simulation
        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        self.density = torch.ones(self.particle_init_position.shape[0]) * density_value

        initial_density, initial_E = self.get_material_params(device)
        mpm_solver.set_E_from_torch(
            mpm_model, initial_E, device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        # setup optimizer and scheduler
        trainable_params = self.E_module.parameters()
        optim_list = [
            {"params": trainable_params, "lr": lr * 1e-10},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_list,
            lr=lr,
            weight_decay=0.0,
        )
        self.trainable_params = trainable_params
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=train_iters,
        )

        # setup train info
        self.step = 0
        self.max_grad_norm = max_grad_norm

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            velocity: [N, 3]
        """
        # init density and young's modulous
        density, youngs_modulus = self.get_material_params(device)
        initial_position_time0 = self.particle_init_position.clone().to(device)

        # init V, F and C
        init_velocity = torch.zeros_like(initial_position_time0, device=device)

        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position_time0.shape[0], dim=0
        )

        particle_C = torch.zeros_like(particle_F, device=device)

        return (
            density,
            youngs_modulus,
            init_velocity,
            particle_F,
            particle_C,
        )

    def get_material_params(self, device):

        current_E = self.E_module.forward()
        current_E = current_E * self.parameter_scale
        # current_E = torch.exp(current_E)
        youngs_modulus = torch.clamp(current_E, 1000.0, 5e8)

        density = self.density.detach().clone()

        return density.to(device), youngs_modulus.to(device)

    def train_one_step(
            self,
            temporal_stride = 3,
            num_substeps = 1,
            loss_decay = 0.95,
            device = "cuda:0"
        ):

        window_size = int(self.window_size_schduler.compute_state(self.step)[0])

        particle_pos = self.particle_init_position.clone()
        # clean grid, stress, F, C and rest initial position
        self.mpm_state.reset_state(
            particle_pos.clone(),
            None,
            None,  # .clone(),
            device=device,
            requires_grad=True,
        )
        self.mpm_state.set_require_grad(True)

        print(f"""step {self.step} : get simulation input..""")
        (
            density,
            youngs_modulus,
            particle_velo,
            particle_F,
            particle_C,
        ) = self.get_simulation_input(device)

        delta_time = self.frame_length
        substep_size = delta_time / num_substeps

        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size

        for start_time_idx in range(0, window_size, temporal_stride):

            end_time_idx = min(start_time_idx + temporal_stride, window_size)

            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)

            print(f"""step {self.step} start_idx {start_time_idx} : get reference...""")
            gt_frame = cv2.imread(
                os.path.join(self.reference_path, f"{end_time_idx-1}.png".rjust(8, "0"))
            )
            gt_frame = torch.from_numpy(cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR), ).permute(2, 0, 1).float()
            gt_frame = (gt_frame / 255).to(device)

            if start_time_idx != 0:
                density, youngs_modulus = self.get_material_params(
                    device
                )
            print(f"""step {self.step} start_idx {start_time_idx} : do forward...""")
            particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
                MPMDifferentiableSimulationClean.apply(
                    self.mpm_solver,
                    self.mpm_state,
                    self.mpm_model,
                    substep_size,
                    num_step_with_grad,
                    particle_pos,
                    particle_velo,
                    particle_F,
                    particle_C,
                    youngs_modulus,
                    density,
                    None,
                    device,
                    True,
                    0,
                )
            )

            # substep-3: render gaussian
            print(f"""step {self.step} start_idx {start_time_idx} : calculate cov and rot...""")
            cov3D, rot = Calculate_Cov_and_Rot.apply(self.init_cov.view(-1), particle_F, device)
            simulated_image = self.stage_renderer.render_image_from_gaussian(particle_pos, cov3D.view(-1, 6), self.opacity, self.shs, rot)
            # print("debug", simulated_video.shape, gt_frame.shape, gaussian_pos.shape, init_xyzs.shape, density.shape, query_mask.sum().item())

            # do the backward calculation
            print(f"""step {self.step} start_idx {start_time_idx} : calculate loss...""")
            l2_loss = 0.5 * F.mse_loss(simulated_image, gt_frame, reduction="mean")
            ssim_loss = compute_ssim(simulated_image, gt_frame)
            loss = l2_loss * (1.0 - self.ssim) + (1.0 - ssim_loss) * self.ssim

            loss = loss * (loss_decay**end_time_idx)

            loss = loss / (end_time_idx - start_time_idx)
            print(f"""step {self.step} start_idx {start_time_idx} : loss is {loss}. do backward..""")
            loss.backward()

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

        # do the stepping
        if (
            self.step % self.gradient_accumulation_steps == 0
            or self.step == (self.train_iters - 1)
        ):

            # torch.nn.utils.clip_grad_norm_(
            #     self.trainable_params,
            #     self.max_grad_norm,
            #     error_if_nonfinite=False,
            # )  # error if nonfinite is false

            self.optimizer.step()
            self.optimizer.zero_grad()
            # with torch.no_grad():
            #     self.E_module.E.data.clamp_(1e-2, 1e7)
        
        self.scheduler.step()
        
        print(f"""step {self.step} : new E is {self.E_module.E}""")


    def train(
            self, 
            temporal_stride = 3,
            num_substeps = 1,
            loss_decay = 0.95,
            device = "cuda:0"
        ):

        # might remove tqdm when multiple node
        for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
            self.train_one_step(
                temporal_stride = temporal_stride,
                num_substeps = num_substeps,
                loss_decay = loss_decay,
                device = device
            )
            self.step += 1