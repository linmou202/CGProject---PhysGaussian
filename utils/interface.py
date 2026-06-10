from typing import Optional, Tuple
from jaxtyping import Float, Int, Shaped
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

import warp as wp

from mpm_solver_warp.warp_utils import from_torch_safe, MyTape, CondTape
from mpm_solver_warp.mpm_solver_diff import MPMWARPDiff
from mpm_solver_warp.mpm_utils import (compute_position_l2_loss, aggregate_grad, compute_posloss_with_grad, 
                                        compute_Closs_with_grad, compute_Floss_with_grad, compute_posloss_with_grad, compute_veloloss_with_grad)
from mpm_solver_warp.mpm_data_structure import MPMStateStruct, MPMModelStruct, get_float_array_product

class MPMDifferentiableSimulationClean(autograd.Function):
    """
    Current version does not support grad for density. 
    Please set vol, mass before calling this function.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: autograd.function.FunctionCtx,
        mpm_solver: MPMWARPDiff,
        mpm_state: MPMStateStruct,
        mpm_model: MPMModelStruct,
        substep_size: float, 
        num_substeps: int,
        particle_x: Float[Tensor, "n 3"], 
        particle_v: Float[Tensor, "n 3"],
        particle_F: Float[Tensor, "n 3 3"],
        particle_C: Float[Tensor, "n 3 3"],
        E: Float[Tensor, "n"],
        particle_density: Optional[Float[Tensor, "n"]]=None,
        query_mask: Optional[Int[Tensor, "n"]] = None,
        device: str="cuda:0",
        requires_grad: bool=True,
        extra_no_grad_steps: int=0,
    ) -> Tuple[Float[Tensor, "n 3"], Float[Tensor, "n 3"], Float[Tensor, "n 9"], Float[Tensor, "n 9"], Float[Tensor, "n 6"]]:
        """
        Args:
            query_mask: [n] 0 or 1.  1 means the density or young's modulus, or poisson'ratio of this particle can change.
        """
        
        # initialization work is done before calling forward! 

        num_particles = particle_x.shape[0]

        mpm_state.continue_from_torch(
            particle_x, particle_v, particle_F, particle_C, device=device, requires_grad=True
        )
        # set x, v, F, C.

        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_E = False
            
        mpm_solver.set_E(mpm_model, E_inp, device=device)
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=query_mask,
            device=device,
            requires_grad=True,
            update_mass=True)
        
        prev_state = mpm_state

        if extra_no_grad_steps > 0:
            with torch.no_grad():
                for i in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=True)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                    prev_state = next_state

        # following steps will be checkpointed. then replayed in backward
        ctx.prev_state = prev_state

        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, requires_grad)
        next_state_list = [] 

        with cond_tape:
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    prev_state.particle_density,
                    prev_state.particle_vol,
                    prev_state.particle_mass,
                ],
                device=device,
            )
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)

            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=True)
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                next_state_list.append(next_state)
                prev_state = next_state
        
        ctx.mpm_solver = mpm_solver
        ctx.mpm_model = mpm_model
        ctx.next_state_list = next_state_list
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.tape = cond_tape.tape

        ctx.save_for_backward(query_mask)

        last_state = next_state
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone()
        particle_velo = wp.to_torch(last_state.particle_v).detach().clone()
        particle_F = wp.to_torch(last_state.particle_F_trial).detach().clone()
        particle_C = wp.to_torch(last_state.particle_C).detach().clone()
        # [N * 6, ]
        particle_cov = wp.to_torch(last_state.particle_cov).detach().clone()

        particle_cov = particle_cov.view(-1, 6)

        return particle_pos, particle_velo, particle_F, particle_C, particle_cov
    

    @staticmethod
    def backward(ctx, out_pos_grad: Float[Tensor, "n 3"], out_velo_grad: Float[Tensor, "n 3"], 
                 out_F_grad: Float[Tensor, "n 9"], out_C_grad: Float[Tensor, "n 9"], out_cov_grad: Float[Tensor, "n 6"]):
        
        num_particles = ctx.num_particles
        device = ctx.device
        mpm_solver, mpm_model = ctx.mpm_solver, ctx.mpm_model
        tape = ctx.tape
        starting_state = ctx.prev_state
        
        next_state_list = ctx.next_state_list
        next_state = next_state_list[-1]

        query_mask = ctx.saved_tensors
    
        with wp.ScopedDevice(device):
            
            grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
            if out_velo_grad is not None:
                grad_velo_wp = from_torch_safe(out_velo_grad, dtype=wp.vec3, requires_grad=False)
            else:
                grad_velo_wp = None
            
            if out_F_grad is not None:
                grad_F_wp = from_torch_safe(out_F_grad, dtype=wp.mat33, requires_grad=False)
            else:
                grad_F_wp = None
            
            if out_C_grad is not None:
                grad_C_wp = from_torch_safe(out_C_grad, dtype=wp.mat33, requires_grad=False)
            else:
                grad_C_wp = None
            
            with tape:
                loss_wp = torch.zeros(1, device=device)
                loss_wp = wp.from_torch(loss_wp, requires_grad=True)
                target_pos_detach = wp.clone(next_state.particle_x, device=device, requires_grad=False)
                wp.launch(
                    compute_posloss_with_grad, 
                    dim=num_particles,
                    inputs=[
                        next_state,
                        target_pos_detach,
                        grad_pos_wp,
                        0.5,
                        loss_wp,
                    ],
                    device=device,
                )
                if grad_velo_wp is not None:
                    target_velo_detach = wp.clone(next_state.particle_v, device=device, requires_grad=False)
                    wp.launch(
                        compute_veloloss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_velo_detach,
                            grad_velo_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,
                    )
                
                if grad_F_wp is not None:
                    target_F_detach = wp.clone(next_state.particle_F_trial, device=device, requires_grad=False)
                    wp.launch(
                        compute_Floss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_F_detach,
                            grad_F_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,
                    )
                if grad_C_wp is not None:
                    target_C_detach = wp.clone(next_state.particle_C, device=device, requires_grad=False)
                    wp.launch(
                        compute_Closs_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_C_detach,
                            grad_C_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,)

            # wp.synchronize_device(device)            
            tape.backward(loss_wp)
            # from IPython import embed; embed()

        pos_grad = wp.to_torch(starting_state.particle_x.grad).detach().clone()
        velo_grad = wp.to_torch(starting_state.particle_v.grad).detach().clone()
        F_grad = wp.to_torch(starting_state.particle_F_trial.grad).detach().clone()
        C_grad = wp.to_torch(starting_state.particle_C.grad).detach().clone()
        # print("debug back", velo_grad)

        # grad for E. TODO: add spatially varying E later
        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[
                    E_grad,
                    mpm_model.E.grad,
                ],
                device=device,
            )
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone()

        # grad for density
        if starting_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(starting_state.particle_density.grad).detach()
        density_mask_grad = None

        tape.zero()
        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), "in sim func")
        # from IPython import embed; embed()
        
        return (None, None, None, None, None,
                pos_grad, velo_grad, F_grad, C_grad, 
                E_grad,
                density_grad, density_mask_grad, 
                None, None, None)


@wp.kernel
def get_cov_from_F(init_cov: wp.array(dtype=float), particle_F: wp.array(dtype=wp.mat33), cur_cov: wp.array(dtype=float)):
    p = wp.tid()
    F = particle_F[p]

    original_cov = wp.mat33(0.0)
    original_cov[0, 0] = init_cov[p * 6]
    original_cov[0, 1] = init_cov[p * 6 + 1]
    original_cov[0, 2] = init_cov[p * 6 + 2]
    original_cov[1, 0] = init_cov[p * 6 + 1]
    original_cov[1, 1] = init_cov[p * 6 + 3]
    original_cov[1, 2] = init_cov[p * 6 + 4]
    original_cov[2, 0] = init_cov[p * 6 + 2]
    original_cov[2, 1] = init_cov[p * 6 + 4]
    original_cov[2, 2] = init_cov[p * 6 + 5]

    cov = F * original_cov * wp.transpose(F)

    cur_cov[p * 6] = cov[0, 0]
    cur_cov[p * 6 + 1] = cov[0, 1]
    cur_cov[p * 6 + 2] = cov[0, 2]
    cur_cov[p * 6 + 3] = cov[1, 1]
    cur_cov[p * 6 + 4] = cov[1, 2]
    cur_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def get_R_from_F(particle_F: wp.array(dtype=wp.mat33), cur_rot: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    F = particle_F[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    cur_rot[p] = wp.transpose(R)


@wp.kernel
def compute_covloss_with_grad(
    cov_vec: wp.array(dtype=float),
    gt_vec: wp.array(dtype=float),
    grad: wp.array(dtype=float),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    l2 =( (cov_vec[6*tid] - (gt_vec[6*tid] - grad[6*tid] * dt))**2.0 + 
          (cov_vec[6*tid + 1] - (gt_vec[6*tid + 1] - grad[6*tid + 1] * dt))**2.0 +
          (cov_vec[6*tid + 2] - (gt_vec[6*tid + 2] - grad[6*tid + 2] * dt))**2.0 +
          (cov_vec[6*tid + 3] - (gt_vec[6*tid + 3] - grad[6*tid + 3] * dt))**2.0 +
          (cov_vec[6*tid + 4] - (gt_vec[6*tid + 4] - grad[6*tid + 4] * dt))**2.0 +
          (cov_vec[6*tid + 5] - (gt_vec[6*tid + 5] - grad[6*tid + 5] * dt))**2.0
        )
    
    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_rotloss_with_grad(
    rot_mat: wp.array(dtype=wp.mat33),
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = rot_mat[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)
    mat_diff = mat_ - mat_gt

    l2 = wp.ddot(mat_diff, mat_diff)
    # l2 = wp.sqrt(
    #     mat_diff[0, 0] ** 2.0
    #     + mat_diff[0, 1] ** 2.0
    #     + mat_diff[0, 2] ** 2.0
    #     + mat_diff[1, 0] ** 2.0
    #     + mat_diff[1, 1] ** 2.0
    #     + mat_diff[1, 2] ** 2.0
    #     + mat_diff[2, 0] ** 2.0
    #     + mat_diff[2, 1] ** 2.0
    #     + mat_diff[2, 2] ** 2.0
    # )

    wp.atomic_add(loss, 0, l2)

class Calculate_Cov_and_Rot(autograd.Function):
    """
    Current version does not support grad for density. 
    Please set vol, mass before calling this function.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: autograd.function.FunctionCtx,
        init_cov: Float[Tensor, "n"],
        particle_F: Float[Tensor, "n 3 3"],
        device
    ) -> Tuple[Float[Tensor, "n"], Float[Tensor, "n 3 3"]]:

        num_particles = particle_F.shape[0]

        # following steps will be checkpointed. then replayed in backward
        init_cov_wp = from_torch_safe(init_cov, dtype=wp.float32, requires_grad=True)
        F_wp = from_torch_safe(particle_F, dtype=wp.mat33, requires_grad=True)
        cur_cov_wp = wp.zeros_like(init_cov_wp) 
        cur_rot_wp = wp.zeros_like(F_wp) 

        wp_tape = MyTape()

        with wp_tape:
            wp.launch(
                kernel=get_cov_from_F,
                dim=num_particles,
                inputs=[
                    init_cov_wp,
                    F_wp,
                    cur_cov_wp
                ],
                device=device,
            )
            wp.launch(
                kernel=get_R_from_F,
                dim=num_particles,
                inputs=[
                    F_wp,
                    cur_rot_wp
                ],
                device=device,
            )
        
        ctx.init_cov_wp = init_cov_wp
        ctx.F_wp = F_wp
        ctx.cur_cov_wp = cur_cov_wp
        ctx.cur_rot_wp = cur_rot_wp
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.tape = wp_tape

        cur_cov = wp.to_torch(cur_cov_wp).detach().clone()
        cur_rot = wp.to_torch(cur_rot_wp).detach().clone()

        return cur_cov, cur_rot
    

    @staticmethod
    def backward(ctx, out_cov_grad: Float[Tensor, "n"], out_rot_grad: Float[Tensor, "n 3 3"]):
        
        init_cov_wp = ctx.init_cov_wp
        F_wp = ctx.F_wp
        cur_cov_wp = ctx.cur_cov_wp
        cur_rot_wp = ctx.cur_rot_wp
        device = ctx.device
        num_particles = ctx.num_particles
        tape = ctx.tape
        
        with wp.ScopedDevice(device):

            if out_cov_grad is not None:
                grad_cov_wp = from_torch_safe(out_cov_grad, wp.float32, requires_grad=False)
            else:
                grad_cov_wp = None
            
            if out_rot_grad is not None:
                grad_rot_wp = from_torch_safe(out_rot_grad, dtype=wp.mat33, requires_grad=False)
            else:
                grad_rot_wp = None
            
            with tape:
                loss_wp = torch.zeros(1, device=device)
                loss_wp = wp.from_torch(loss_wp, requires_grad=True)             
                if grad_cov_wp is not None:
                    target_cov_detach = wp.clone(cur_cov_wp, device=device, requires_grad=False)
                    wp.launch(
                        compute_covloss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            cur_cov_wp,
                            target_cov_detach,
                            grad_cov_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,
                    )
                if grad_rot_wp is not None:
                    target_rot_detach = wp.clone(cur_rot_wp, device=device, requires_grad=False)
                    wp.launch(
                        compute_rotloss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            cur_rot_wp,
                            target_rot_detach,
                            grad_rot_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,)

            # wp.synchronize_device(device)            
            tape.backward(loss_wp)
            # from IPython import embed; embed()

        F_grad = wp.to_torch(F_wp.grad).detach().clone()
        # print("debug back", velo_grad)

        tape.zero()
        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), "in sim func")
        # from IPython import embed; embed()
        
        return (None, F_grad, None)