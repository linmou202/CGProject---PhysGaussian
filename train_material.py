import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import point_cloud_utils as pcu

import numpy as np
import logging
import argparse
import shutil
import torch
import os
from time import time
from PIL import Image
import imageio
import numpy as np

# from motionrep.utils.torch_utils import get_sync_time
from einops import rearrange, repeat

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
    MPMDifferentiableSimulationWCheckpoint,
    MPMDifferentiableSimulationClean,
)

from utils.render_utils import *
from utils.train_utils import Young_Moudulous_Map

class Trainer:
    def __init__(self, args):
        self.args = args

        self.ssim = args.ssim
        args.warmup_step = int(args.warmup_step * args.gradient_accumulation_steps)
        args.train_iters = int(args.train_iters * args.gradient_accumulation_steps)

        # setup the gaussians

        self.num_frames = int(args.num_frames)
        self.window_size_schduler = LinearStepAnneal(
            args.train_iters,
            start_state=[args.start_window_size],
            end_state=[13],
            plateau_iters=-1,
            warmup_step=20,
        )

        self.train_iters = args.train_iters

        # init traiable params
        E_nu_list = self.init_trainable_params()
        for p in E_nu_list:
            p.requires_grad = True
        self.E_nu_list = E_nu_list

        self.setup_simulation(dataset_dir, grid_size=args.grid_size)

        trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
        optim_list = [
            {"params": self.E_nu_list, "lr": args.lr * 1e-10},
            {
                "params": self.sim_fields.parameters(),
                "lr": args.lr,
                "weight_decay": 1e-4,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optim_list,
            lr=args.lr,
            weight_decay=0.0,
        )
        self.trainable_params = trainable_params
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        self.sim_fields, self.optimizer, self.scheduler = accelerator.prepare(
            self.sim_fields, self.optimizer, self.scheduler
        )

        # setup train info
        self.step = 0
        self.batch_size = args.batch_size
        self.tv_loss_weight = args.tv_loss_weight

        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.max_grad_norm = args.max_grad_norm

    def init_trainable_params(
        self,
    ):

        # init young modulus and poisson ratio

        young_numpy = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3))).astype(
            np.float32
        )

        young_numpy = np.array([1e5]).astype(np.float32)

        young_modulus = torch.tensor(young_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        poisson_numpy = np.random.uniform(0.1, 0.4)
        poisson_ratio = torch.tensor(poisson_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        trainable_params = [young_modulus, poisson_ratio]

        print(
            "init young modulus: ",
            young_modulus.item(),
            "poisson ratio: ",
            poisson_ratio.item(),
        )
        return trainable_params

    def setup_simulation(self, dataset_dir, grid_size=100):

        device = "cuda:{}".format(self.accelerator.process_index)

        xyzs = self.render_params.gaussians.get_xyz.detach().clone()
        sim_xyzs = xyzs
        sim_cov = (
            self.render_params.gaussians.get_covariance()
            .detach()
            .clone()
        )

        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)

        # filled
        filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")

        if os.path.exists(filled_in_points_path):
            fill_xyzs = pcu.load_mesh_v(filled_in_points_path)  # [n, 3]
            fill_xyzs = fill_xyzs[
                np.random.choice(
                    fill_xyzs.shape[0], int(fill_xyzs.shape[0] * 0.25), replace=False
                )
            ]
            fill_xyzs = torch.from_numpy(fill_xyzs).float().to("cuda")
            self.fill_xyzs = fill_xyzs
            print(
                "loaded {} internal filled points from: ".format(fill_xyzs.shape[0]),
                filled_in_points_path,
            )
        else:
            self.fill_xyzs = None

        if self.fill_xyzs is not None:
            render_mask_in_sim_pts = torch.cat(
                [
                    torch.ones_like(sim_xyzs[:, 0]).bool(),
                    torch.zeros_like(fill_xyzs[:, 0]).bool(),
                ],
                dim=0,
            ).to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill_xyzs], dim=0)
            sim_cov = torch.cat(
                [sim_cov, sim_cov.new_ones((fill_xyzs.shape[0], sim_cov.shape[-1]))],
                dim=0,
            )
            self.render_mask = render_mask_in_sim_pts
        else:
            self.render_mask = torch.ones_like(sim_xyzs[:, 0]).bool().to(device)

        sim_xyzs = (sim_xyzs + shift) / scale

        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        # point cloud resample with kmeans

        downsample_scale = self.args.downsample_scale
        num_cluster = int(sim_xyzs.shape[0] * downsample_scale)
        sim_xyzs = downsample_with_kmeans_gpu(sim_xyzs, num_cluster)

        sim_gaussian_pos = self.render_params.gaussians.get_xyz.detach().clone()
        sim_gaussian_pos = (sim_gaussian_pos + shift) / scale

        cdist = torch.cdist(sim_gaussian_pos, sim_xyzs) * -1.0
        _, top_k_index = torch.topk(cdist, self.args.top_k, dim=-1)
        self.top_k_index = top_k_index

        print("Downsampled to: ", sim_xyzs.shape[0], "by", downsample_scale)

        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())

        num_particles = sim_xyzs.shape[0]

        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(num_particles, device=device, requires_grad=True)

        self.particle_init_position = sim_xyzs.clone()

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            sim_cov,
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=1.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(num_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=1.0, device=device)

        material_params = {
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 2000,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # 0.999,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]
        mpm_solver = MPMWARPDiff(
            num_particles, n_grid=grid_size, grid_lim=1.0, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        # setup boundary condition:
        moving_pts_path = os.path.join(dataset_dir, "moving_part_points.ply")
        if os.path.exists(moving_pts_path):
            moving_pts = pcu.load_mesh_v(moving_pts_path)
            moving_pts = torch.from_numpy(moving_pts).float().to(device)
            moving_pts = (moving_pts + shift) / scale
            freeze_mask = find_far_points(
                sim_xyzs, moving_pts, thres=0.5 / grid_size
            ).bool()
            freeze_pts = sim_xyzs[freeze_mask, :]

            grid_freeze_mask = apply_grid_bc_w_freeze_pts(
                grid_size, 1.0, freeze_pts, mpm_solver
            )
            self.freeze_mask = freeze_mask

            # does not prefer boundary condition on particle
            # freeze_mask_select = setup_boundary_condition_with_points(sim_xyzs, moving_pts,
            #                                                         self.mpm_solver, self.mpm_state, thres=0.5 / grid_size)
            # self.freeze_mask = freeze_mask_select.bool()
        else:
            raise NotImplementedError

        num_freeze_pts = self.freeze_mask.sum()
        print(
            "num freeze pts in total",
            num_freeze_pts.item(),
            "num moving pts",
            num_particles - num_freeze_pts.item(),
        )

        # init fields for simulation, e.g. density, external force, etc.

        # padd init density, youngs,
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.E_nu_list[0].detach()
        )
        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * 0.3

        # load stem for higher density
        stem_pts_path = os.path.join(dataset_dir, "stem_points.ply")
        if os.path.exists(stem_pts_path):
            stem_pts = pcu.load_mesh_v(stem_pts_path)
            stem_pts = torch.from_numpy(stem_pts).float().to(device)
            stem_pts = (stem_pts + shift) / scale
            no_stem_mask = find_far_points(
                sim_xyzs, stem_pts, thres=2.0 / grid_size
            ).bool()
            stem_mask = torch.logical_not(no_stem_mask)
            density[stem_mask] = 2000
            print("num stem pts", stem_mask.sum().item())

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        self.sim_fields.train()

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        # scaling
        velocity = velocity * 0.1  # not padded yet
        ret_velocity = torch.zeros_like(initial_position_time0)
        ret_velocity[query_mask, :] = velocity

        # init F, and C

        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position_time0.shape[0], dim=0
        )
        particle_C = torch.zeros_like(particle_F)

        return (
            density,
            youngs_modulus,
            ret_poisson,
            ret_velocity,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        )

    def get_material_params(self, device):

        initial_position_time0 = self.particle_init_position.detach()

        # query_mask = torch.logical_not(self.freeze_mask)
        query_mask = torch.ones_like(self.freeze_mask).bool()
        query_pts = initial_position_time0[query_mask, :]
        if self.args.entropy_cls > 0:
            sim_params, entropy = self.sim_fields(query_pts)
        else:
            sim_params = self.sim_fields(query_pts)
            entropy = torch.zeros(1).to(sim_params.device)

        sim_params = sim_params * 1000
        # sim_params = torch.exp(self.sim_fields(query_pts))

        # density = sim_params[..., 0]

        youngs_modulus = self.young_modulus.detach().clone()
        youngs_modulus[query_mask] += sim_params[..., 0]

        # young_modulus = torch.exp(sim_params[..., 0]) + init_young
        youngs_modulus = torch.clamp(youngs_modulus, 1000.0, 5e8)

        density = self.density.detach().clone()
        # density[self.freeze_mask] = 100000
        ret_poisson = self.poisson_ratio.detach().clone()

        return density, youngs_modulus, ret_poisson, entropy

    def train_one_step(self):

        self.sim_fields.train()
        device = "cuda:{}".format(accelerator.process_index) #TODO: please pass device as input
        cam = data["cam"][0]

        gt_videos = data["video_clip"][0, 1 : self.num_frames, ...]

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

        (
            density,
            youngs_modulus,
            poisson,
            particle_velo,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_simulation_input(device)

        init_velo_mean = particle_velo[query_mask, :].mean().item()

        num_particles = particle_pos.shape[0]

        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        checkpoint_steps = self.args.checkpoint_steps

        start_time_idx = max(0, window_size - self.args.compute_window)

        temporal_stride = self.args.stride

        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size

        for start_time_idx in range(0, window_size, temporal_stride):

            end_time_idx = min(start_time_idx + temporal_stride, window_size)

            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)

            gt_frame = gt_videos[[end_time_idx - 1]]

            if start_time_idx != 0:
                density, youngs_modulus, poisson, entropy = self.get_material_params(
                    device
                )

            if checkpoint_steps > 0 and checkpoint_steps < num_step_with_grad:
                for time_step in range(0, num_step_with_grad, checkpoint_steps):
                    num_step = min(num_step_with_grad - time_step, checkpoint_steps)
                    if num_step == 0:
                        break
                    particle_pos, particle_velo, particle_F, particle_C = (
                        MPMDifferentiableSimulationWCheckpoint.apply(
                            self.mpm_solver,
                            self.mpm_state,
                            self.mpm_model,
                            substep_size,
                            num_step,
                            particle_pos,
                            particle_velo,
                            particle_F,
                            particle_C,
                            youngs_modulus,
                            self.E_nu_list[1],
                            density,
                            query_mask,
                            device,
                            True,
                            0,
                        )
                    )
            else:
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
                        self.E_nu_list[1],
                        density,
                        query_mask,
                        device,
                        True,
                        0,
                    )
                )

            # substep-3: render gaussian

            gaussian_pos = particle_pos * self.scale - self.shift
            undeformed_gaussian_pos = (
                self.particle_init_position * self.scale - self.shift
            )
            disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
            # gaussian_pos.requires_grad = True

            simulated_video = render_gaussian_seq_w_mask_with_disp(
                cam,
                self.render_params,
                undeformed_gaussian_pos.detach(),
                self.top_k_index,
                [disp_offset],
                self.sim_mask_in_raw_gaussian, # since we don't take out the far points, please change this to torch.ones/zeros_like...
            )

            # print("debug", simulated_video.shape, gt_frame.shape, gaussian_pos.shape, init_xyzs.shape, density.shape, query_mask.sum().item())

            l2_loss = 0.5 * F.mse_loss(simulated_video, gt_frame, reduction="mean")
            ssim_loss = compute_ssim(simulated_video, gt_frame)
            loss = l2_loss * (1.0 - self.ssim) + (1.0 - ssim_loss) * self.ssim

            loss = loss * (self.args.loss_decay**end_time_idx)

            loss = loss / self.args.compute_window
            loss.backward()

            # from IPython import embed; embed()
            # print(self.E_nu_list[1].grad)

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

        nu_grad_norm = self.E_nu_list[1].grad.norm(2).item()
        spatial_grad_norm = 0
        for p in self.sim_fields.parameters():
            if p.grad is not None:
                spatial_grad_norm += p.grad.norm(2).item()
        velo_grad_norm = 0

        if (
            self.step % self.gradient_accumulation_steps == 0
            or self.step == (self.train_iters - 1)
            or (self.step % self.log_iters == self.log_iters - 1)
        ):

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
                error_if_nonfinite=False,
            )  # error if nonfinite is false

            self.optimizer.step()
            self.optimizer.zero_grad()
            with torch.no_grad():
                self.E_nu_list[0].data.clamp_(1e-1, 1e8)
                self.E_nu_list[1].data.clamp_(1e-2, 0.449)
        
        self.scheduler.step()

        print(
            "nu: ",
            self.E_nu_list[1].item(),
            nu_grad_norm,
            spatial_grad_norm,
            velo_grad_norm,
            "young_mean, max:",
            youngs_modulus.mean().item(),
            youngs_modulus.max().item(),
            do_velo_opt,
            "init_velo_mean:",
            init_velo_mean,
        )


    def train(self):
        # might remove tqdm when multiple node
        for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
            self.train_one_step()
            if self.step % self.log_iters == self.log_iters - 1:
                if self.accelerator.is_main_process:
                    self.save()
                    # self.test()
            # self.accelerator.wait_for_everyone()
            self.step += 1

def setup_trainer(
    stage_renderer,
    ssim, # the ratio between L1 loss and ssim loss
    gradient_accumulate_steps,  # after how many backwards we step once


):
    # torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer(stage_renderer)
    trainer.train()
