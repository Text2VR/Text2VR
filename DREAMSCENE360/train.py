'''
docker-compose run --rm dreamscene360 \
  micromamba run -n dev \
  python train.py \
    -s "/workspace/DREAMSCENE360/data/living_room_woSink" \
    -m "/workspace/DREAMSCENE360/output/living_room_woSink_ply" \
    --pano_path "/workspace/DREAMSCENE360/data/living_room_woSink/inpainted_panorama.png" \
    --no_perturb_loss \
    --iterations 7000 \
    --test_iterations 7000 \
    --save_iterations 5000 7000
'''


#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from utils.feature_extractor import get_Feature_from_DinoV2
from random import randint
from utils.loss_utils import l1_loss, ssim, cosine_similarity_loss
from gaussian_renderer import render, network_gui
from torchmetrics.functional.regression import pearson_corrcoef
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import matplotlib.pyplot as plt
from scene import Scene, GaussianModel
from utils.depth_utils import estimate_depth

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



# === add near imports ===
import math
import torch.nn.functional as F
from utils.graphics_utils import fov2focal

# --- helpers -------------------------------------------------
def _get_hw(cam):
    """Return (H, W) robustly from Camera."""
    if hasattr(cam, "image_height") and hasattr(cam, "image_width"):
        return int(cam.image_height), int(cam.image_width)
    if hasattr(cam, "height") and hasattr(cam, "width"):
        return int(cam.height), int(cam.width)
    # fall back to tensor shape of original_image: [C,H,W]
    return int(cam.original_image.shape[-2]), int(cam.original_image.shape[-1])

def _get_fov(cam):
    """Return (FoVx, FoVy) in radians; handle different attribute names."""
    fovx = getattr(cam, "FoVx", getattr(cam, "FovX", None))
    fovy = getattr(cam, "FoVy", getattr(cam, "FovY", None))
    if fovx is None or fovy is None:
        raise AttributeError("Camera is missing FoVx/FovX or FoVy/FovY.")
    return float(fovx), float(fovy)

def _view_dirs_cam(h, w, fx, fy, cx, cy, device, dtype):
    """Make per-pixel camera-space unit rays [H,W,3]."""
    jj, ii = torch.meshgrid(torch.arange(w, device=device),
                            torch.arange(h, device=device),
                            indexing='xy')
    x = (jj.to(dtype) - cx) / fx
    y = (ii.to(dtype) - cy) / fy
    z = torch.ones_like(x, dtype=dtype)
    d = torch.stack([x, y, z], dim=-1)
    return d / (d.norm(dim=-1, keepdim=True) + 1e-8)
# -------------------------------------------------------------

def plane_depth_for_view(cam, plane, device):
    """
    Compute z-depth of a world-space plane for the given view.
    plane: (n, d) where n is world normal, d is scalar in n·x + d = 0.
    cam.R must be world<-cam (R_wc).
    Returns: [H, W] z-depth in camera frame (torch).
    """
    dtype = torch.float32
    H, W = _get_hw(cam)
    FoVx, FoVy = _get_fov(cam)

    fx = torch.tensor(fov2focal(FoVx, W), device=device, dtype=dtype)
    fy = torch.tensor(fov2focal(FoVy, H), device=device, dtype=dtype)
    cx = torch.tensor(0.5 * W, device=device, dtype=dtype)
    cy = torch.tensor(0.5 * H, device=device, dtype=dtype)

    d_cam = _view_dirs_cam(H, W, fx, fy, cx, cy, device, dtype)          # [H,W,3]
    R_wc = torch.tensor(cam.R, device=device, dtype=dtype)               # [3,3]
    d_w  = torch.einsum('ij,hwj->hwi', R_wc, d_cam)                      # [H,W,3]

    n, d = plane
    n = torch.tensor(n, device=device, dtype=dtype)                      # [3]
    denom = (d_w @ n).clamp_min(1e-6)                                    # [H,W]
    s = (-float(d)) / denom                                              # ray length to plane
    z_plane = s * d_cam[..., 2]                                          # convert to z-depth
    return z_plane

def bottom_mask_for_view(cam, theta_deg=30, device='cuda'):
    """
    Build a boolean mask for “downward” rays in world coords (nadir band).
    world up is +Z; select rays with d_world.z < -sin(theta).
    """
    dtype = torch.float32
    H, W = _get_hw(cam)
    FoVx, FoVy = _get_fov(cam)

    fx = torch.tensor(fov2focal(FoVx, W), device=device, dtype=dtype)
    fy = torch.tensor(fov2focal(FoVy, H), device=device, dtype=dtype)
    cx = torch.tensor(0.5 * W, device=device, dtype=dtype)
    cy = torch.tensor(0.5 * H, device=device, dtype=dtype)

    d_cam = _view_dirs_cam(H, W, fx, fy, cx, cy, device, dtype)          # [H,W,3]
    R_wc = torch.tensor(cam.R, device=device, dtype=dtype)
    d_w  = torch.einsum('ij,hwj->hwi', R_wc, d_cam)                      # [H,W,3]

    th = math.radians(theta_deg)
    return d_w[..., 2] < -math.sin(th)


def floor_normal_loss(z, band_ratio=0.18):
    """
    Encourage the nadir band to have near-up normals (flat floor).
    Uses Sobel gradients with padding to preserve shape.
    """
    if z.dim() == 3 and z.shape[0] == 1:
        z = z[0]                           # [H,W]

    # Sobel kernels (shape-preserving via padding)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      device=z.device, dtype=z.dtype).view(1,1,3,3) / 8.0
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                      device=z.device, dtype=z.dtype).view(1,1,3,3) / 8.0
    z4d = z[None,None,...]                 # [1,1,H,W]
    gx = F.conv2d(z4d, kx, padding=1)[0,0] # [H,W]
    gy = F.conv2d(z4d, ky, padding=1)[0,0] # [H,W]

    # Approx surface normal = [-dx, -dy, 1], normalized
    nx = -gx
    ny = -gy
    nz = torch.ones_like(z)
    n = torch.stack([nx, ny, nz], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-6)

    # Use only bottom band (nadir)
    H = z.shape[0]
    hband = max(1, int(band_ratio * H))
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask[-hband:, :] = True

    # Align normals with +z (up): minimize (1 - |nz|)
    return (1.0 - n[..., 2].abs()[mask]).mean()


# === add util funcs (top-level) ===
def ssi_depth_loss(pred_z, gt_z, mask=None, eps=1e-6):
    if mask is None:
        mask = torch.isfinite(gt_z)
    p = pred_z[mask].reshape(-1)
    g = gt_z[mask].reshape(-1)
    if p.numel() == 0:
        return torch.tensor(0.0, device=pred_z.device)
    # scale & shift align: argmin_{a,b} || a p + b - g ||^2
    a = torch.std(g) / (torch.std(p) + eps)
    b = torch.mean(g) - a * torch.mean(p)
    p_aligned = a * pred_z + b
    return (p_aligned[mask] - gt_z[mask]).abs().mean()

def grad_alignment_loss(pred_z, gt_z):
    # edge-aware: Sobel-like finite differences
    def dxdy(x):
        dx = x[:, :, 1:] - x[:, :, :-1]
        dy = x[:, 1:, :] - x[:, :-1, :]
        return dx, dy
    px, py = dxdy(pred_z[None])
    gx, gy = dxdy(gt_z[None])
    return (px - gx).abs().mean() + (py - gy).abs().mean()




# ★★★ MODIFICATION 1: Add 'no_perturb_loss' to the function signature ★★★
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             api_key, self_refinement, num_prompt, max_rounds, pano_path=None, no_perturb_loss=False):
    
    if pano_path is not None:
        setattr(dataset, "pano_path", pano_path)
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, api_key, self_refinement, num_prompt, max_rounds)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    perturbation_viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # rendered_depth = render_pkg["depth"]
        # gt_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        # === replace depth loss block in the main loop ===
        rendered_depth = render_pkg["depth"]
        # --- ADD: force to [H, W] and dtype align ---
        if rendered_depth.dim() == 3 and rendered_depth.shape[0] == 1:
            rendered_depth = rendered_depth[0]
        gt_depth = torch.tensor(viewpoint_cam.depth_image, device=image.device, dtype=rendered_depth.dtype)
        # --------------------------------------------
        # gt_depth = torch.tensor(viewpoint_cam.depth_image).to(image.device)  # z-depth

        # old:
        # depth_weight = 0.05
        # loss_depth = depth_weight * (1 - pearson_corrcoef(rendered_depth.reshape(-1,1)[:,0], -gt_depth.reshape(-1,1)[:,0]))
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_weight * loss_depth

        # new:
        depth_weight = 0.1
        loss_depth_main = ssi_depth_loss(rendered_depth, gt_depth) + 0.2 * grad_alignment_loss(rendered_depth, gt_depth)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_weight * loss_depth_main
        
        # --- Floor-plane hinge prior (keeps floor from sagging) ---
        if hasattr(scene, "floor_plane") and scene.floor_plane is not None:
            # where you compute the floor hinge loss
            z_plane = plane_depth_for_view(viewpoint_cam, scene.floor_plane, image.device)
            bmask  = bottom_mask_for_view(viewpoint_cam, theta_deg=55, device=image.device)  # was 30
            margin = 0.03 * z_plane[bmask].mean().detach()  # was 0.01
            hinge  = torch.relu(rendered_depth - (z_plane - margin))
            floor_w = 0.30  # was 0.15
            loss = loss + floor_w * hinge[bmask].mean()
            def floor_laplacian(z):
                k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=z.device, dtype=z.dtype).view(1,1,3,3)
                z4d = z[None,None,...]
                return torch.abs(F.conv2d(z4d, k, padding=1))[0,0]
            # add after hinge:
            smooth_w = 0.05
            loss += smooth_w * floor_laplacian(rendered_depth)[bmask].mean()
        # ----------------------------------------------------------
        # === end of replace ===

        loss_feature = torch.tensor(0).cuda() 
        loss_perturbation_depth = torch.tensor(0).cuda() 

        # ★★★ MODIFICATION 2: Wrap the advanced training block with the new flag ★★★
        # This entire block will only run if --no_perturb_loss is NOT specified.
        if not no_perturb_loss and iteration > 5400:
            if iteration > 5400 and iteration <= 6600:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=1).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration > 6600 and iteration <= 7800:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=2).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration <= 9000:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=3).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))

            perturbation_render_pkg = render(perturbation_viewpoint_cam, gaussians, pipe, bg)
            perturbation_image, perturbation_rendered_depth= perturbation_render_pkg["render"], perturbation_render_pkg["depth"]
            # --- ADD: force to [H, W] and dtype align ---
            if perturbation_rendered_depth.dim() == 3 and perturbation_rendered_depth.shape[0] == 1:
                perturbation_rendered_depth = perturbation_rendered_depth[0]
            # --------------------------------------------
            
            # === fix perturbation block ===
            # old bugged lines:
            # pred_depth = estimate_depth(perturbation_image)
            # loss_perturbation_depth = (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1)[:, 0], - gt_depth.reshape(-1, 1)[:, 0]))
            # if torch.isnan(loss_perturbation_depth).sum() == 0:
            #     loss += depth_weight * loss_perturbation_depth

            # correct:
            pred_depth = estimate_depth(perturbation_image).to(image.device)     # estimated z-depth(or if inverse -> transform z inside)
            loss_perturbation_depth = ssi_depth_loss(perturbation_rendered_depth, pred_depth) \
                                    + 0.2 * grad_alignment_loss(perturbation_rendered_depth, pred_depth)

            if torch.isnan(loss_perturbation_depth).sum() == 0:
                loss += 0.5 * depth_weight * loss_perturbation_depth
            loss += 0.5 * floor_w * floor_normal_loss(perturbation_rendered_depth)
            # === end of fix perturbation block ===

            pred_feature = get_Feature_from_DinoV2(perturbation_image)
            ref_image = perturbation_viewpoint_cam.original_image.cuda()
            ref_feature = get_Feature_from_DinoV2(ref_image)
            loss_feature = cosine_similarity_loss(pred_feature, ref_feature)
            
            feature_loss_weight = 0.05
            loss += feature_loss_weight * loss_feature 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth_main, loss, l1_loss, loss_perturbation_depth, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                print(f"\n[EXPORTING FOR UNITY] Saving .ply file for iteration {iteration}...")
                ply_path = os.path.join(scene.model_path, f"point_cloud_iter_{iteration}.ply")
                scene.gaussians.save_ply(ply_path)
                print(f"-> Saved: {ply_path}")

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/feature_loss', loss_feature.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_perturbation_depth', loss_perturbation_depth.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test = 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 9_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 7_000, 9_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--self_refinement", action='store_true', default=False)
    parser.add_argument("--num_prompt", type=int, default = 3)
    parser.add_argument("--max_rounds", type=int, default = 3)
    parser.add_argument("--pano_path", type=str, default=None, help="Optional path to an equirectangular panorama to use directly.")
    
    # ★★★ MODIFICATION 3: Add the new command-line argument ★★★
    parser.add_argument("--no_perturb_loss", action='store_true', default=False, help="Disable the perturbation loss stage (after 5400 iterations) for faster training.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # ★★★ MODIFICATION 4: Pass the new argument to the training function ★★★
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.api_key,
        args.self_refinement, args.num_prompt, args.max_rounds,
        pano_path=args.pano_path,
        no_perturb_loss=args.no_perturb_loss
    )
    
    print("\nTraining complete.")
'''
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
#sys.path.append('Depth-Anything-TorchVersion')
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from utils.feature_extractor import get_Feature_from_DinoV2
from random import randint
from utils.loss_utils import l1_loss, ssim, cosine_similarity_loss
from gaussian_renderer import render, network_gui
from torchmetrics.functional.regression import pearson_corrcoef
import sys
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np ###
import matplotlib.pyplot as plt ###
from scene import Scene, GaussianModel
### midas ###
from utils.depth_utils import estimate_depth
#############

# ### depth anything ###
# sys.path.append(os.path.join(os.path.abspath("."), "Depth-Anything-TorchVersion"))
# from depth_anything.dpt import DepthAnything 
# from estimate_depth import depth_anything ###
# ######################

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             api_key, self_refinement, num_prompt, max_rounds, pano_path=None):
    
    if pano_path is not None:
        setattr(dataset, "pano_path", pano_path)
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, api_key, self_refinement, num_prompt, max_rounds)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    perturbation_viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg["depth"] ###
        gt_depth = torch.tensor(viewpoint_cam.depth_image).cuda() ###
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        depth_weight = 0.05 #0.005  
        loss_depth = depth_weight * (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1)[:, 0], - gt_depth.reshape(-1, 1)[:, 0]))
        
        loss =  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_weight * loss_depth
        loss_feature = torch.tensor(0).cuda() 
        loss_perturbation_depth = torch.tensor(0).cuda() 

        if iteration > 5400:
            if iteration > 5400 and iteration <= 6600:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=1).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration > 6600 and iteration <= 7800:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=2).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration <= 9000:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=3).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))

            perturbation_render_pkg = render(perturbation_viewpoint_cam, gaussians, pipe, bg)
            perturbation_image, perturbation_rendered_depth= perturbation_render_pkg["render"], perturbation_render_pkg["depth"]
            
            ### perturbation depth loss
            pred_depth = estimate_depth(perturbation_image)
            loss_perturbation_depth =   (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1)[:, 0], - gt_depth.reshape(-1, 1)[:, 0]))

            if torch.isnan(loss_perturbation_depth).sum() == 0:
                loss += depth_weight * loss_perturbation_depth
            
            
            ### feature loss
            pred_feature = get_Feature_from_DinoV2(perturbation_image)# (1, 768)
            ref_image = perturbation_viewpoint_cam.original_image.cuda()
            ref_feature = get_Feature_from_DinoV2(ref_image)
            loss_feature = cosine_similarity_loss(pred_feature, ref_feature)
            
            feature_loss_weight = 0.05
            loss += feature_loss_weight * loss_feature 

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) ###
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
                ############################# EXPORTING FOR UNITY #############################
                print(f"\n[EXPORTING FOR UNITY] Saving .ply file for iteration {iteration}...")
                ply_path = os.path.join(scene.model_path, f"point_cloud_iter_{iteration}.ply")
                scene.gaussians.save_ply(ply_path)
                print(f"-> Saved: {ply_path}")
                ###############################################################################

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/feature_loss', loss_feature.item(), iteration) ###
        tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth.item(), iteration) ###
        tb_writer.add_scalar('train_loss_patches/loss_perturbation_depth', loss_perturbation_depth.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 9_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 9_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--self_refinement", action='store_true', default=False)
    parser.add_argument("--num_prompt", type=int, default = 3)
    parser.add_argument("--max_rounds", type=int, default = 3)
    parser.add_argument("--pano_path", type=str, default=None, help="Optional path to an equirectangular panorama to use directly (skip text generation).")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.api_key,
        args.self_refinement, args.num_prompt, args.max_rounds,
        pano_path=args.pano_path
    )
    # All done
    print("\nTraining complete.")
'''