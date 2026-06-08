from typing import Optional, Tuple
from jaxtyping import Float, Int, Shaped
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

import warp as wp

from mpm_solver_warp.warp_utils import from_torch_safe, MyTape, CondTape
from mpm_solver_warp.mpm_solver_diff import MPMWARPDiff
from mpm_solver_warp.mpm_utils import compute_position_l2_loss, aggregate_grad, compute_posloss_with_grad
from mpm_solver_warp.mpm_data_structure import MPMStateStruct, MPMModelStruct, get_float_array_product
from mpm_solver_warp.mpm_utils import (compute_Closs_with_grad, compute_Floss_with_grad, 
                                                compute_posloss_with_grad, compute_veloloss_with_grad)

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
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
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