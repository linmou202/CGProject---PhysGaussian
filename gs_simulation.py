import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
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

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.video_utils import *
from utils.vlm_utils import *

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    if rotated_pos.shape[0] == 0:
        raise RuntimeError("There's nothing to simulate!")

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    # clustering is postponed until now because only the points that need to be simulated need to be clustered.
    # PART1_DONE: use DBSCAN to cluster the points. the last tensor stores the non-clustered points
    cls_pos, cls_opacity, cls_cov, cls_screen_points, cls_shs, cls_size = DBSCAN_cluster(
        transformed_pos,
        init_opacity,
        init_cov,
        init_screen_points,
        init_shs
    )
    num_items = len(cls_pos) - 1
    assert num_items >= 0
    cluster_index = []
    for i in range (0, num_items):
        cluster_index.append(None)
        cluster_index.append(cls_pos[i].shape[0])

    # PART2_TODO: add and change some code here to cooperate with function "generate_bounded_image".
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )
    vlm_camera = get_camera_view(
        model_path,
        default_camera_index=camera_params["default_camera_index"],
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        show_hint=False,
        init_azimuthm=camera_params["init_azimuthm"],
        init_elevation=camera_params["init_elevation"],
        init_radius=camera_params["init_radius"],
        move_camera=False,
        current_frame=0,
        delta_a=camera_params["delta_a"],
        delta_e=camera_params["delta_e"],
        delta_r=camera_params["delta_r"],
    )
    vlm_rasterize = initialize_resterize(vlm_camera, gaussians, pipeline, background)
    vlm_render_pos = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(transformed_pos), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )
    vlm_render_cov = init_cov / (scale_origin * scale_origin)
    vlm_render_cov = apply_inverse_cov_rotations(vlm_render_cov, rotation_matrices)
    vlm_render_opacity = init_opacity
    vlm_render_shs = init_shs
    vlm_colors = convert_SH(vlm_render_shs, vlm_camera, gaussians, vlm_render_pos)
    vlm_means2d = torch.zeros_like(
        vlm_render_pos, dtype=vlm_render_pos.dtype, requires_grad=True, device="cuda"
    )
    vlm_rendering, _ = vlm_rasterize(
        means3D=vlm_render_pos,
        means2D=vlm_means2d,
        shs=None,
        colors_precomp=vlm_colors,
        opacities=vlm_render_opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=vlm_render_cov,
    )
    vlm_image = vlm_rendering.permute(1, 2, 0).detach().cpu().numpy()
    vlm_image = cv2.cvtColor(vlm_image, cv2.COLOR_RGB2BGR)
    vlm_image = np.clip(vlm_image * 255.0, 0, 255).astype(np.uint8)

    world_clusters = [
        apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(cls_pos[i]), scale_origin, original_mean_pos
            ),
            rotation_matrices,
        )
        for i in range(num_items)
    ]
    vlm_boxes = []
    image_width = int(vlm_camera.image_width)
    image_height = int(vlm_camera.image_height)
    for i, cluster in enumerate(world_clusters):
        if cluster.shape[0] == 0:
            continue
        ones = torch.ones((cluster.shape[0], 1), device=cluster.device, dtype=cluster.dtype)
        hom_points = torch.cat([cluster[:, :3], ones], dim=1)
        clip_points = hom_points @ vlm_camera.full_proj_transform
        w = clip_points[:, 3:4]
        valid = torch.isfinite(clip_points).all(dim=1) & (torch.abs(w[:, 0]) > 1e-7)
        if not torch.any(valid):
            continue
        ndc = clip_points[valid, :3] / w[valid]
        ndc = ndc[ndc[:, 2] >= 0]
        if ndc.shape[0] == 0:
            continue
        x = (ndc[:, 0] + 1.0) * 0.5 * (image_width - 1)
        y = (1.0 - ndc[:, 1]) * 0.5 * (image_height - 1)
        xy = torch.stack([x, y], dim=1).detach().cpu().numpy()
        finite_mask = np.isfinite(xy).all(axis=1)
        xy = xy[finite_mask]
        if xy.shape[0] == 0:
            continue
        lower = np.min(xy, axis=0) - 8
        upper = np.max(xy, axis=0) + 8
        vlm_boxes.append(
            {
                "label": i,
                "left_bottom": lower.tolist(),
                "right_top": upper.tolist(),
            }
        )
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    vlm_name = config_stem[:-7] if config_stem.endswith("_config") else config_stem
    bounded_image_path = os.path.join(
        "generated_data", "bounded_image", f"{vlm_name}_bounded.png"
    )
    generate_bounded_image(
        vlm_image,
        vlm_boxes,
        output_path=bounded_image_path,
        coordinate_origin="top_left",
    )
    print(f"Generated bounded image: {bounded_image_path}")
    call_vlm(bounded_image_path, os.path.join("generated_data", "vlm_data", f"{vlm_name}.json"))
    cls_E, cls_filling_method = get_initial_params(vlm_name)

    # ensure the data is correctly generated
    # assert num_items == cls_E.shape[0]

    if cls_filling_method is None or filling_params.get("use_vlm", False):
        filling_methods = resolve_filling_methods(
            filling_params.get("methods", None),
            filling_params.get("method", "legacy"),
            len(cls_pos),
        )
    
    cluster_sizes = [int(size.item()) for size in cls_size]
    cluster_budgets = allocate_filling_budgets(
        cluster_sizes,
        filling_params["max_particles_num"],
        filling_params["min_particles_num_per_object"],
    )

    # FRAMEWORK_DONE: fill the items one by one
    cls_mpm_init_pos = []
    num_particles = 0
    if filling_params is not None:
        for i in range(0, num_items):
            print(str.encode(f"""Filling internal particles of item {i} ..."""))
            cls_mpm_init_pos.append(fill_particles(
                pos=cls_pos[i],
                opacity=cls_opacity[i],
                cov=cls_cov[i],
                grid_n=filling_params["n_grid"],
                max_samples=cluster_budgets[i],
                grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
                density_thres=filling_params["density_threshold"],
                search_thres=filling_params["search_threshold"],
                max_particles_per_cell=filling_params["max_partciels_per_cell"],
                search_exclude_dir=filling_params["search_exclude_direction"],
                ray_cast_dir=filling_params["ray_cast_direction"],
                boundary=filling_params["boundary"],
                smooth=filling_params["smooth"],
                method=cls_filling_method[i],
                sample_ratio=filling_params.get("mcis_sample_ratio", None),
                mcis_sigma=filling_params.get("mcis_sigma", 0.02),
            ).to(device=device))

            cluster_index[2*i] = num_particles
            cluster_index[2*i + 1] = num_particles + cluster_index[2*i + 1]
            num_particles += cls_mpm_init_pos[i].shape[0]
        
        cls_mpm_init_pos.append(transformed_pos[num_items].to(device=device))
        cluster_index.append(num_particles)
        num_particles += cls_mpm_init_pos[num_items].shape[0]
        cluster_index.append(num_particles)

        if args.debug:
            particle_position_tensor_list_to_ply(cls_mpm_init_pos, "./log/filled_particles.ply")
        
    else:
        for i in range(0, num_items+1):
            cls_mpm_init_pos.append(transformed_pos[i].to(device=device))

            cluster_index[2*i] = num_particles
            cluster_index[2*i + 1] = num_particles + cluster_index[2*i + 1]
            num_particles += cls_mpm_init_pos[i].shape[0]

    # FRAMEWORK_DONE: concat the position list
    mpm_init_pos = torch.cat(cls_mpm_init_pos, dim=0)

    # FRAMEWORK_DONE: concat the tensor in the lists after initializing them
    # init the mpm parameters
    print("Initializing MPM solver and setting up boundary conditions...")

    if filling_params is not None and filling_params["visualize"] == True:
        for i in range(0, num_items):
            cls_shs[i], cls_opacity[i], cls_cov[i] = init_filled_particles(
                mpm_init_pos[cluster_index[2*i]:cluster_index[2*i+1]],
                cls_shs[i],
                cls_cov[i],
                cls_opacity[i],
                mpm_init_pos[cluster_index[2*i+1]:cluster_index[2*i+2]],
            )
        mpm_init_cov = torch.cat(cls_cov, dim=0)
        opacity = torch.cat(cls_opacity, dim=0)
        shs = torch.cat(cls_shs, dim=0)
        gs_num = num_particles
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        shs = torch.zeros((mpm_init_pos.shape[0], init_shs.shape[1]), device=device)
        opacity = torch.zeros((mpm_init_pos.shape[0]), device=device)

        for i in range(0, num_items+1):
            mpm_init_cov[cluster_index[2*i]:cluster_index[2*i + 1]] = cls_cov[i]
            shs[cluster_index[2*i]:cluster_index[2*i + 1]] = cls_cov[i]
            opacity[cluster_index[2*i]:cluster_index[2*i + 1]] = cls_opacity[i]
        
        gs_num = num_particles

    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)
    
    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # FRAMEWORK_TODO: Change things below this line ---------------------------------
    # build inverted index
    inverted_index = torch.zeros((mpm_init_pos.shape[0]), dtype=torch.int8, device=device)
    current_item = 0
    for i in range(0, gs_num):
        if (current_item < num_items and i >= cluster_index[current_item*2]):
            current_item = current_item + 1
        inverted_index = current_item
    
    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(cluster_index, inverted_index, cls_E, material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )
