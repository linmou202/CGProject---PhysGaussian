import argparse
import os
import gc
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


@torch.no_grad()
def downsample_particles_with_kmeans(
        points,
        num_clusters,
        max_iter=8,
        max_pairs_per_chunk=8000000,
        seed=0,
):
    num_points = points.shape[0]
    num_clusters = int(num_clusters)
    if num_clusters <= 0 or num_clusters >= num_points:
        labels = torch.arange(num_points, dtype=torch.long, device=points.device)
        return points.detach().clone(), labels

    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    init_index = torch.randperm(num_points, generator=generator, device=points.device)[:num_clusters]
    centers = points.detach()[init_index].clone()
    labels = torch.empty(num_points, dtype=torch.long, device=points.device)
    chunk_size = max(1, min(num_points, int(max_pairs_per_chunk) // num_clusters))

    print(
        f"=> kmeans downsample simulation particles from {num_points} to {num_clusters}, "
        f"chunk_size={chunk_size}, iters={max_iter}"
    )
    for iter_idx in range(int(max_iter)):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, dtype=points.dtype, device=points.device)

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            dist = torch.cdist(points[start:end].detach(), centers)
            chunk_labels = torch.argmin(dist, dim=1)
            labels[start:end] = chunk_labels
            sums.index_add_(0, chunk_labels, points[start:end].detach())
            counts.index_add_(0, chunk_labels, torch.ones(end - start, dtype=points.dtype, device=points.device))

        non_empty = counts > 0
        centers[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(-1)
        if torch.any(~non_empty):
            empty_num = int((~non_empty).sum().item())
            refill_index = torch.randperm(num_points, generator=generator, device=points.device)[:empty_num]
            centers[~non_empty] = points.detach()[refill_index]
        print(f"=> kmeans iter {iter_idx + 1}/{max_iter}, empty_clusters={int((~non_empty).sum().item())}")

    return centers, labels


@torch.no_grad()
def average_by_labels(values, labels, num_labels):
    flat_values = values.detach().reshape(values.shape[0], -1)
    sums = torch.zeros((num_labels, flat_values.shape[1]), dtype=flat_values.dtype, device=flat_values.device)
    counts = torch.zeros(num_labels, dtype=flat_values.dtype, device=flat_values.device)
    sums.index_add_(0, labels, flat_values)
    counts.index_add_(0, labels, torch.ones(labels.shape[0], dtype=flat_values.dtype, device=flat_values.device))
    counts = torch.clamp(counts, min=1.0)
    averaged = sums / counts.unsqueeze(-1)
    return averaged.reshape((num_labels, *values.shape[1:]))


@torch.no_grad()
def majority_by_labels(values, labels, num_labels):
    values = values.detach().long()
    num_classes = int(values.max().item()) + 1
    combined = labels.long() * num_classes + values
    counts = torch.bincount(combined, minlength=num_labels * num_classes)
    counts = counts.reshape(num_labels, num_classes)
    return torch.argmax(counts, dim=1).to(values.device)


@torch.no_grad()
def compute_drive_neighbor_weights(
        query_points,
        drive_points,
        topk=8,
        max_pairs_per_chunk=8000000,
):
    topk = min(int(topk), drive_points.shape[0])
    chunk_size = max(1, min(query_points.shape[0], int(max_pairs_per_chunk) // drive_points.shape[0]))
    all_indices = []
    all_weights = []
    print(
        f"=> computing {topk}-NN interpolation weights for {query_points.shape[0]} points "
        f"from {drive_points.shape[0]} driving particles"
    )
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        dist = torch.cdist(query_points[start:end].detach(), drive_points.detach())
        values, indices = torch.topk(dist, k=topk, dim=1, largest=False)
        weights = 1.0 / torch.clamp(values, min=1e-8)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        all_indices.append(indices)
        all_weights.append(weights)
    return torch.cat(all_indices, dim=0), torch.cat(all_weights, dim=0)


def interpolate_from_driving_particles(
        query_origin,
        drive_origin,
        drive_pos,
        drive_F,
        neighbor_index,
        neighbor_weight,
):
    drive_disp = drive_pos - drive_origin
    query_disp = (drive_disp[neighbor_index] * neighbor_weight.unsqueeze(-1)).sum(dim=1)
    query_pos = query_origin + query_disp
    query_F = (drive_F[neighbor_index] * neighbor_weight.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    return query_pos, query_F


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
            device = "cuda:0",
            render_init_pos = None,
            render_init_cov = None,
            render_shs = None,
            render_opacity = None,
            drive_origin_pos = None,
            drive_neighbor_index = None,
            drive_neighbor_weight = None,
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
        self.init_cov = init_cov.detach()
        self.opacity = opacity.detach()
        self.shs = shs.detach()
        self.render_init_pos = render_init_pos
        self.render_init_cov = render_init_cov
        self.render_shs = render_shs.detach() if render_shs is not None else None
        self.render_opacity = render_opacity.detach() if render_opacity is not None else None
        self.drive_origin_pos = drive_origin_pos
        self.drive_neighbor_index = drive_neighbor_index
        self.drive_neighbor_weight = drive_neighbor_weight

        self.num_frames = int(num_frames)
        self.frame_length = frame_length
        self.window_size_schduler = LinearStepAnneal(
            train_iters,
            start_state=[int(num_frames / 4) + 1],
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
            num_step_without_grad = 0

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
                    num_step_without_grad,
                )
            )

            # substep-3: render gaussian
            print(f"""step {self.step} start_idx {start_time_idx} : calculate cov and rot...""")
            if self.drive_neighbor_index is not None:
                render_pos, render_F = interpolate_from_driving_particles(
                    self.render_init_pos.to(device),
                    self.drive_origin_pos.to(device),
                    particle_pos,
                    particle_F,
                    self.drive_neighbor_index.to(device),
                    self.drive_neighbor_weight.to(device),
                )
                cov3D, rot = Calculate_Cov_and_Rot.apply(self.render_init_cov.view(-1).to(device), render_F, device)
                simulated_image = self.stage_renderer.render_image_from_gaussian(
                    render_pos,
                    cov3D.view(-1, 6),
                    self.render_opacity,
                    self.render_shs,
                    rot,
                )
            else:
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
            particle_cov = particle_cov.detach()
            del loss, l2_loss, ssim_loss, simulated_image, gt_frame, cov3D, rot, particle_cov
            if 'render_pos' in locals():
                del render_pos
            if 'render_F' in locals():
                del render_F
            wp.synchronize_device(device)
            torch.cuda.empty_cache()
            gc.collect()

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
