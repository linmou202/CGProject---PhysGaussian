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
from train_material import *

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
from mpm_solver_warp.mpm_data_structure import *
from mpm_solver_warp.mpm_solver_diff import MPMWARPDiff
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

ti.init(arch=ti.cuda, device_memory_GB=7.0)


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
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_iters", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=float, default=5)
    parser.add_argument("--temporal_stride", type=float, default=2)
    parser.add_argument("--loss_decay", type=float, default=0.95)
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
    if not os.path.exists(args.ref_path):
        AssertionError("Reference Images does not exist!")
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

    print("Applying transformations...")
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
    print("Clustering point clouds...")
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
    generate_bounded_image('bounded_image')
    call_vlm('bounded_image','generated_data')
    cls_E, cls_filling_method = get_initial_params('generated_data')
    if cls_E is None:
        cls_E = torch.tensor(material_params["E"])

    # ensure the data is correctly generated
    # assert num_items == cls_E.shape[0]

    # FRAMEWORK_DONE: fill the items one by one
    cls_mpm_init_pos = []
    num_particles = 0
    if filling_params is not None:
        
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

        for i in range(0, num_items):
            print(f"""Filling internal particles of item {i} ...""")
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
                # method=cls_filling_method[i],
                # sample_ratio=filling_params.get("mcis_sample_ratio", None),
                # mcis_sigma=filling_params.get("mcis_sigma", 0.02),
            ).to(device=device))

            cluster_index[2*i] = num_particles
            cluster_index[2*i + 1] = num_particles + cluster_index[2*i + 1]
            num_particles += cls_mpm_init_pos[i].shape[0]
        
        cls_mpm_init_pos.append(cls_pos[num_items].to(device=device))
        cluster_index.append(num_particles)
        num_particles += cls_mpm_init_pos[num_items].shape[0]
        cluster_index.append(num_particles)

        if args.debug:
            particle_position_tensor_list_to_ply(cls_mpm_init_pos, "./log/filled_particles.ply")
        
    else:
        # create index space for the non-clustered points
        cluster_index.append(None)
        cluster_index.append(None)
        for i in range(0, num_items+1):
            cls_mpm_init_pos.append(cls_pos[i].to(device=device))

            cluster_index[2*i] = num_particles
            cluster_index[2*i + 1] = num_particles + cluster_index[2*i + 1]
            num_particles += cls_mpm_init_pos[i].shape[0]
    
    # FRAMEWORK_DONE: concat the tensor in the lists after initializing them
    # init the mpm inputs
    mpm_init_pos = torch.cat(cls_mpm_init_pos, dim=0)

    # build inverted index and filling mask
    inverted_index = torch.zeros((mpm_init_pos.shape[0]), dtype=torch.int8, device=device)
    original_mask = torch.zeros((mpm_init_pos.shape[0]), dtype=torch.bool)
    current_section = 0
    for i in range(0, num_particles):
        while (current_section < (2*num_items + 1) and i >= cluster_index[current_section + 1]):
            current_section = current_section + 1
            print(f"""building mask for section {current_section}""")
        inverted_index[i] = current_section // 2
        if current_section % 2 == 0 or filling_params["visualize"] == True:
            original_mask[i] = True
        else:
            original_mask[i] = False

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
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], *init_cov.shape[1:]), device=device)
        shs = torch.zeros((mpm_init_pos.shape[0], *init_shs.shape[1:]), device=device)
        opacity = torch.zeros((mpm_init_pos.shape[0], *init_opacity.shape[1:]), device=device)

        for i in range(0, num_items+1):
            mpm_init_cov[cluster_index[2*i]:cluster_index[2*i + 1]] = cls_cov[i]
            shs[cluster_index[2*i]:cluster_index[2*i + 1]] = cls_shs[i]
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

    # FRAMEWORK_DONE: set stage renderer
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

    stage_renderer = STAGERENDERER(
        preprocessing_params["sim_area"],
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
        init_screen_points,
        original_mask
    )
    stage_renderer.set_rasterizer(0)

    # FRAMEWORK_TODO: Change things below this line ----------------------------------------
    
    
    # set up the mpm solver for training
    print("Initializing MPM solver for the training...")

    mpm_state = MPMStateStruct()
    mpm_state.init(num_particles, device=device, requires_grad=True)
    mpm_state.from_torch(
        mpm_init_pos.to(device).clone(),
        mpm_init_vol.float().to(device).clone(),
        mpm_init_cov.to(device).clone(),
        device=device,
        requires_grad=True,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )

    mpm_model = MPMModelStruct()
    mpm_model.init(num_particles, device=device, requires_grad=True)
    mpm_model.init_other_params(n_grid=material_params["n_grid"], grid_lim=material_params["grid_lim"], device=device)
    
    mpm_solver = MPMWARPDiff(
        num_particles, n_grid=material_params["n_grid"], grid_lim=material_params["grid_lim"], device=device
    )
    mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, mpm_state, bc_params, time_params)

    # do the training
    print("starting to train...")
    E_items = torch.ones(num_items+1) * cls_E.item() / 1000
    trainer = Trainer(
        stage_renderer,
        mpm_init_cov,
        shs,
        opacity,
        mpm_state,
        mpm_model,
        mpm_solver,
        E_items,
        material_params["density"],
        inverted_index,
        args.ref_path,
        frame_length = time_params["frame_dt"],
        num_frames = args.num_frames,
        ssim = 0.9,                        # the ratio between L1 loss and ssim loss
        parameter_scale = 1000,            # the scaling of E
        max_grad_norm = 1.0,
        gradient_accumulation_steps = 1,   # after how many backwards we step once
        warmup_step = args.warmup_steps,
        train_iters = args.train_iters,
        lr = args.lr,
        device = device
    )

    trainer.train(
        temporal_stride = args.temporal_stride,
        num_substeps = int(time_params["frame_dt"] / time_params["substep_dt"]),
        loss_decay = args.loss_decay,
        device = device
    )
    learnt_density, learnt_E = trainer.get_material_params(device)

    # set up the mpm solver for simulation
    print("Initializing MPM solver for the simulation...")

    mpm_state = MPMStateStruct()
    mpm_state.init(num_particles, device=device, requires_grad=True)
    mpm_state.from_torch(
        mpm_init_pos.to(device).clone(),
        mpm_init_vol.float().to(device).clone(),
        mpm_init_cov.to(device).clone(),
        device=device,
        requires_grad=True,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )

    mpm_model = MPMModelStruct()
    mpm_model.init(num_particles, device=device, requires_grad=True)
    mpm_model.init_other_params(n_grid=material_params["n_grid"], grid_lim=material_params["grid_lim"], device=device)
    
    mpm_solver = MPMWARPDiff(
        num_particles, n_grid=material_params["n_grid"], grid_lim=material_params["grid_lim"], device=device
    )
    mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)
    mpm_solver.set_E_from_torch(mpm_model, learnt_E, device=device)

    set_boundary_conditions(mpm_solver, mpm_state, bc_params, time_params)
    mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

    # run the simulation
    print("starting the simulation...")
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            mpm_state,
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
        if (camera_params["move_camera"]):
            stage_renderer.set_rasterizer(frame)

        for step in range(step_per_frame):
            mpm_solver.p2g2p(mpm_model, mpm_state, frame, substep_dt, device=device)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                mpm_state,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch(mpm_state, mpm_model).to(device)
            particle_F = mpm_solver.export_particle_F_to_torch(mpm_state, mpm_model).to(device)
            cov3D, rot = Calculate_Cov_and_Rot.apply(mpm_init_cov.to(device), particle_F, device)
            cov3D = cov3D.view(-1, 6).to(device)
            rot = rot.view(-1, 3, 3).to(device)

            rendering = stage_renderer.render_image_from_gaussian(pos, cov3D, opacity_render, shs_render, rot)
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
                print()
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
    print("work done.")
