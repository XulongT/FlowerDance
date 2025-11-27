import copy
import os
import pickle
from pathlib import Path
from functools import partial
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from einops import rearrange, reduce, repeat
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

from torch.autograd.functional import jvp
class FlowMatching(nn.Module):
    def __init__(self, model, horizon, repr_dim, smpl, sigma=0.1, loss_type="l2"):
        super().__init__()
        self.horizon = horizon
        self.repr_dim = repr_dim
        self.model = model
        self.smpl = smpl
        self.sigma = sigma
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

        # EMA
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

        # p2 weighting (continuous version)
        self.alpha = 2
        self.jvp_fn = jvp

    def sample_two_timesteps(self, b, device):

        t = torch.rand((b, 1, 1), device=device)
        r = torch.rand((b, 1, 1), device=device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        return t, r

    def fm_losses(self, x0, cond, genre, mode="Uniform"):

        b, device = x0.shape[0], x0.device
        e = torch.randn_like(x0)
        w = 1


        t, r = self.sample_two_timesteps(b, device)
        x_t = (1 - t) * x0 + t * e
        v = e - x0

        def u_func(x_t, t, r):
            h = t - r
            return self.model(x_t, cond, genre, t.view(-1), h.view(-1))

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        u_pred, dudt = self.jvp_fn(u_func, (x_t, t, r), (v, dtdt, drdt))
        u_tgt = (v - (t - r) * dudt).detach()

        # flow matching loss
        # fm_loss = (u_pred - u_tgt) ** 2
        # fm_loss = fm_loss.sum(dim=(1, 2))
        fm_loss = self.loss_fn(u_pred, u_tgt, reduction="none")
        fm_loss = reduce(fm_loss, "b ... -> b (...)", "mean")
        # adp_wt = (fm_loss.detach() + 1e-3) ** 0.75
        fm_loss = fm_loss / ((fm_loss.detach() + 1e-3) ** 0.75)

        r1 = torch.zeros_like(t)
        t1 = torch.rand((b, 1, 1), device=x0.device)  
        e = torch.randn_like(x0)


        x_t = (1 - t1) * x0 + t1 * e


        u = self.model(x_t, cond, genre, t1.view(-1), t1.view(-1))


        model_out = x_t - t1 * u 
        target = x0

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * w

        # # split off contact from the rest
        model_contact, model_out = torch.split(model_out, (4, model_out.shape[2] - 4), dim=2)
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)
        
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)
        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * w

        # velocity loss
        target_xp_v = target_xp[:, 1:] - target_xp[:, :-1]
        model_xp_v = model_xp[:, 1:] - model_xp[:, :-1]
        v_loss = self.loss_fn(model_xp_v, target_xp_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * w

        # foot skate loss
        # foot_idx = [7, 8, 10, 11]
        # static_idx = model_contact > 0.95  # N x S x 4
        # model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        # model_foot_v = torch.zeros_like(model_feet)
        # model_foot_v[:, :-1] = (model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :])  # (N, S-1, 4, 3)
        # model_foot_v[~static_idx] = 0
        # foot_loss = self.loss_fn(model_foot_v, torch.zeros_like(model_foot_v), reduction="none")
        # foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean") * w

        foot_loss = torch.tensor([0.0])
        # v_loss = torch.tensor([0.0])
        # disp_loss = torch.tensor([0.0])

        losses = (fm_loss.mean(), 0.636 * loss.mean(), 0.323 * v_loss.mean(), 0.646 * fk_loss.mean(), 10.942 * foot_loss.mean(),)
        return sum(losses), losses

    def forward(self, x, cond, genre, t_override=None):
        return self.fm_losses(x, cond, genre)



    @torch.no_grad()
    def euler_sample(self, shape, cond, genre, n_steps=21):
        b, device = shape[0], cond.device
        x = torch.randn(shape, device=device) 

        t_values = torch.linspace(1, 0, n_steps, device=device)
        dt = 1.0 / (n_steps - 1)  
        for i in tqdm(range(n_steps - 1)):
            t = torch.full((b,), t_values[i], device=device)
            h = torch.full((b,), t_values[i] - t_values[i+1], device=device)
            v = self.model.infer_pred(x, cond, genre, t, h)
            x = x - v * dt
        return x

    @torch.no_grad()
    def render_sample(self, shape, cond, genre, normalizer, epoch, render_out, fk_out=None, name=None, sound=True, n_steps=21):
        start = time.perf_counter()
        samples = self.euler_sample(shape, cond, genre, n_steps=n_steps).detach().cpu()
        print('gt', cond.shape, shape, 'pred', samples.shape)
        end = time.perf_counter()
        print(f"Time: {(end - start):.2f} s")


        samples = normalizer.unnormalize(samples)
        sample_contact, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)

        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        q = ax_from_6v(q).to(cond.device)

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = sample_contact.detach().cpu().numpy() if sample_contact is not None else None

        Path(fk_out).mkdir(parents=True, exist_ok=True)
        for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            audioname = os.path.join(*pathparts)
            os.makedirs(os.path.join(fk_out, str(epoch)), exist_ok=True)
            outname = f"{epoch}/{pathparts[-1][:-4]}.pkl"
            pickle.dump({"smpl_poses": qq.reshape((-1, 72)).cpu().numpy(), "smpl_trans": pos_.cpu().numpy(), "full_pose": pose,}, open(f"{fk_out}/{outname}", "wb"))


    @torch.no_grad()
    def inpaint(self, x, mask, cond, genre, n_steps=101):

        b, device = x.shape[0], x.device
        self.model.eval()

        e = torch.randn_like(x)
        x_t = e.clone()


        t_values = torch.linspace(1.0, 0, n_steps, device=device)
        for i in range(len(t_values) - 1):
            t_cur = t_values[i]
            t_next = t_values[i + 1]

            t_batch = torch.full((b,), t_cur, device=device)
            h_batch = torch.full((b,), t_cur - t_next, device=device)  # interval


            v_pred = self.model.infer_pred(x_t, cond, genre, t_batch, h_batch)


            dt = (t_next - t_cur).item()  
            x_t = x_t + v_pred * dt

            mask1 = mask * t_next
            # mask1 = mask
            x_t = mask1 * ((1 - t_next) * x + t_next * e) + (1 - mask1) * x_t

        return x_t


    @torch.no_grad()
    def inpaint_sample(self, x, cond, genre, normalizer, epoch, render_out, fk_out=None, name=None, sound=True, n_steps=101, mode="middle"):

        b, seq, dim = x.shape
        device = x.device
        keep = 300  

        mask = torch.ones((b, seq, dim), device=device)
        if mode == "front":
            mask[:, keep:, :] = 0
        elif mode == "back":
            mask[:, :seq-keep, :] = 0
        elif mode == "middle":
            mask[:, keep:seq-keep, :] = 0
        else:
            raise ValueError(f"Unknown inpaint mode {mode}")


        start = time.perf_counter()
        samples = self.inpaint(x, mask, cond, genre, n_steps=n_steps).detach().cpu()
        print(f"Inpaint mode={mode}, shape={samples.shape}")
        end = time.perf_counter()
        print(f"Inpaint Time: {(end - start):.2f} s")


        samples = normalizer.unnormalize(samples)
        sample_contact, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)


        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        q = ax_from_6v(q).to(cond.device)

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = sample_contact.detach().cpu().numpy() if sample_contact is not None else None


        Path(fk_out).mkdir(parents=True, exist_ok=True)
        for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            audioname = os.path.join(*pathparts)
            os.makedirs(os.path.join(fk_out, str(epoch)), exist_ok=True)
            outname = f"{epoch}/{pathparts[-1][:-4]}.pkl"
            pickle.dump({"smpl_poses": qq.reshape((-1, 72)).cpu().numpy(), "smpl_trans": pos_.cpu().numpy(), "full_pose": pose,}, open(f"{fk_out}/{outname}", "wb"))

