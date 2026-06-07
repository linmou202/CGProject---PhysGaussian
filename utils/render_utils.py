import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torchvision
import torch
import os
import numpy as np
import json
import copy
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

from utils.camera_view_utils import get_camera_view
from utils.transformation_utils import *

def initialize_resterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterize = GaussianRasterizer(raster_settings=raster_settings)
    return rasterize


def load_params_from_gs(
    pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }


def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp

class STAGERENDERER:
    def __init__(
        self,
        sim_area,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
        gaussians,
        pipeline,
        background,
        model_path,
        camera_params,
        viewpoint_center_worldspace,
        observant_coordinates,
        unselected_pos,
        unselected_cov,
        unselected_opacity,
        unselected_shs,
        init_screen_points
    ):
        self.sim_area = sim_area
        self.rotation_matrices = rotation_matrices
        self.scale_origin = scale_origin
        self.original_mean_pos = original_mean_pos


        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        
        self.model_path = model_path
        self.camera_params = camera_params
        self.viewpoint_center_worldspace = viewpoint_center_worldspace
        self.observant_coordinates = observant_coordinates

        self.unselected_pos = unselected_pos
        self.unselected_cov = unselected_cov
        self.unselected_opacity = unselected_opacity
        self.unselected_shs = unselected_shs
        self.init_screen_points = init_screen_points
    
    def set_rasterizer(self, frame_seq):
        self.current_camera = get_camera_view(
            self.model_path,
            default_camera_index=self.camera_params["default_camera_index"],
            center_view_world_space=self.viewpoint_center_worldspace,
            observant_coordinates=self.observant_coordinates,
            show_hint=self.camera_params["show_hint"],
            init_azimuthm=self.camera_params["init_azimuthm"],
            init_elevation=self.camera_params["init_elevation"],
            init_radius=self.camera_params["init_radius"],
            move_camera=self.camera_params["move_camera"],
            current_frame=frame_seq,
            delta_a=self.camera_params["delta_a"],
            delta_e=self.camera_params["delta_e"],
            delta_r=self.camera_params["delta_r"],
        )
        self.rasterize = initialize_resterize(
            self.current_camera, self.gaussians, self.pipeline, self.background
        )
    
    def undo_transform_to_gaussians(self, pos, cov3D):

        pos = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), self.scale_origin, self.original_mean_pos
            ),
            self.rotation_matrices,
        )
        
        cov3D = cov3D / (self.scale_origin * self.scale_origin)
        cov3D = apply_inverse_cov_rotations(cov3D, self.rotation_matrices)

        return pos, cov3D

    def render_image_from_gaussian(
        self,
        pos,
        cov3D,
        opacity,
        shs,
        rot
    ):

        pos = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), self.scale_origin, self.original_mean_pos
            ),
            self.rotation_matrices,
        )
        cov3D = cov3D / (self.scale_origin * self.scale_origin)
        cov3D = apply_inverse_cov_rotations(cov3D, self.rotation_matrices)

        if self.sim_area is not None:
            pos = torch.cat([pos, self.unselected_pos], dim=0)
            cov3D = torch.cat([cov3D, self.unselected_cov], dim=0)
            opacity = torch.cat([opacity, self.unselected_opacity], dim=0)
            shs = torch.cat([shs, self.unselected_shs], dim=0)

        colors_precomp = convert_SH(shs, self.current_camera, self.gaussians, pos, rot)
        rendering, raddi = self.rasterize(
            means3D=pos,
            means2D=self.init_screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
        )

        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        return cv2_img