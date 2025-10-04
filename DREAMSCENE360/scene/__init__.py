# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
import json
import random
import importlib

import numpy as np                     # === NEW: used for rotations and pi
import torch                            # === NEW: used for depth warping and tensors
import cv2 as cv
from PIL import Image
import trimesh

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, SceneInfo     ###
from scene.gaussian_model import GaussianModel, BasicPointCloud                     ###
from arguments import ModelParams
from utils.camera_utils import (
    cameraList_from_camInfos, camera_to_JSON, img_coord_to_pano_direction,
    cam_rays_cam_space, direction_to_pano_coord, pano_to_img_coord,                 ###
    img_coord_from_hw                                                               ###
)
from geo_predictors.pano_geo_predictor import *
from utils.utils import read_image
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.save_data import save_data

# Allow optional text->pano generator on demand
sys.path.append('stitch_diffusion/kohya_trainer')


# -----------------------------------------------------------------------------
# === NEW: helper to add small rotational jitter (no translation) for "perturbation" views
# This avoids fake parallax from single-ERP training which causes bowl-shaped floors/ceilings.
# -----------------------------------------------------------------------------
def _jitter_R(R, yaw_deg=2.0, pitch_deg=1.0):
    yaw = (np.random.rand() - 0.5) * 2 * np.deg2rad(yaw_deg)
    pit = (np.random.rand() - 0.5) * 2 * np.deg2rad(pitch_deg)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0          ],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(pit),-np.sin(pit)],
                   [0, np.sin(pit), np.cos(pit)]], dtype=np.float32)
    return (Ry @ Rx @ R).astype(np.float32)


# -----------------------------------------------------------------------------
# === NEW: convert ERP ray-length s(θ,φ) to z-depth for a given pinhole view
# Inputs:
#   dist_erp: [H_erp, W_erp] torch, ray-length along direction (not z)
#   R_wc:     3x3 rotation world <- cam, numpy or torch
#   fx,fy,cx,cy: intrinsics of the perspective view
#   w,h:      perspective size (width,height)
# Output:
#   z: [h,w] torch, per-pixel z-depth in the camera frame
# -----------------------------------------------------------------------------
def erp_raylen_to_view_z(dist_erp, R_wc, fx, fy, cx, cy, w, h):
    device = dist_erp.device
    # Pixel grid
    jj, ii = torch.meshgrid(torch.arange(w, device=device),
                            torch.arange(h, device=device),
                            indexing='xy')
    x = (jj - cx) / fx
    y = (ii - cy) / fy
    z = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)                                  # [h,w,3]
    d_cam = d_cam / (d_cam.norm(dim=-1, keepdim=True) + 1e-8)

    # Cam->World
    if not torch.is_tensor(R_wc):
        R_wc = torch.from_numpy(np.asarray(R_wc)).to(device=device, dtype=torch.float32)
    d_world = torch.einsum('ij,hwj->hwi', R_wc, d_cam)                       # [h,w,3]

    # World dir -> ERP coords (normalized to [-1,1])
    lam = torch.atan2(d_world[..., 2], d_world[..., 0])                      # [-pi, pi]
    phi = torch.asin(torch.clamp(d_world[..., 1], -1, 1))                    # [-pi/2, pi/2]
    u_norm = lam / np.pi                                                     # [-1,1]
    v_norm = -2.0 * phi / np.pi                                             # [-1,1] top is -1

    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)                # [1,h,w,2]
    # Sample ray-length s from ERP
    s = torch.nn.functional.grid_sample(
        dist_erp[None, None, ...], grid, mode='bilinear',
        padding_mode='border', align_corners=True
    )[0, 0]                                                                  # [h,w]

    # Convert to z-depth
    z = s * d_cam[..., 2].clamp_min(1e-6)
    return z


# -----------------------------------------------------------------------------
# === CHANGED: remove arbitrary scaling of distances in PCD construction
# The previous code scaled distances by 0.7*max, which distorts global geometry.
# Here we keep the original ray-length scale from the predictor.
# -----------------------------------------------------------------------------
def pcd_from_depths(pano_img, distances, height, width, source_path):
    """
    Build point cloud from ERP ray-length and ray directions.
    pano_img:  [H,W,3] torch, RGB in [0,1]
    distances: [H,W]   torch, ray-length along unit directions
    """
    pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()  # [H,W,3]
    pts = pano_dirs * distances.squeeze()[..., None]                                   # no rescaling
    pts = pts.cpu().numpy().reshape(-1, 3)
    return pts


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


# -----------------------------------------------------------------------------
# === CHANGED: add `is_eval` argument to avoid using undefined global `eval`
# Build CameraInfo lists and seed depth images per view with proper z-depth
# -----------------------------------------------------------------------------
def get_info_from_params(source_path, pano_img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts, is_eval=False):
    H, W, _ = pano_img.shape
    n_pers, _, h, w = pers_imgs.shape

    cam_infos_unsorted = []
    cam_perturbation_infos_unsorted = []
    cam_perturbation_infos_unsorted_stage2 = []
    cam_perturbation_infos_unsorted_stage3 = []

    # We will attach per-view z-depth converted from ERP ray-length.
    dist_erp = distances.squeeze().contiguous()  # [H,W] torch

    for i in range(n_pers):
        with torch.no_grad():
            # Prepare RGB image for CameraInfo
            img = pers_imgs[i].cpu().numpy()                  # C,H,W in [0,1]
            img = img.transpose(1, 2, 0)                      # H,W,C
            img = (img * 255).astype('uint8')
            img = Image.fromarray(img)

            intri = {
                'fx': float(fx[i].item()),
                'fy': float(fy[i].item()),
                'cx': float(cx[i].item()),
                'cy': float(cy[i].item()),
            }
            fovx = focal2fov(intri['fx'], w)
            fovy = focal2fov(intri['fy'], h)

            # rot_w2c is world->cam. We need R_wc = (rot_w2c)^T for cam->world.
            R_wc = np.transpose(np.asarray(rot_w2c[i].cpu(), dtype=np.float32))    # [3,3]
            R = R_wc                                                                # CameraInfo expects R (row-major)
            T = np.transpose(np.array([0, 0, 0], dtype=np.float32))                 # === CHANGED: keep translation zero

            # === NEW: compute z-depth for this perspective view from ERP distances
            z_depth = erp_raylen_to_view_z(
                dist_erp, R_wc,
                intri['fx'], intri['fy'], intri['cx'], intri['cy'], w, h
            )
            z_depth_np = z_depth.cpu().numpy().astype(np.float32)                   # [h,w]

            uid = i
            image_name = f'image{i}'
            try:
                os.makedirs(os.path.join(source_path, 'images'), exist_ok=True)
            except Exception:
                pass
            image_path = os.path.join(source_path, 'images', image_name)

            # Base training view
            cam_info = CameraInfo(
                uid=uid, R=R, T=T, FovY=fovy, FovX=fovx, image=img,
                image_path=image_path, image_name=image_name, width=w, height=h,
                depth=z_depth_np                                                    # === NEW: pass z-depth to downstream
            )
            # cam_info.depth = z_depth_np                                             # === NEW: pass z-depth to downstream
            cam_infos_unsorted.append(cam_info)

            # --- Perturbation views: rotation-only jitter, zero translation ---
            R1 = _jitter_R(R_wc, yaw_deg=2.0, pitch_deg=1.0)
            R2 = _jitter_R(R_wc, yaw_deg=3.0, pitch_deg=2.0)
            R3 = _jitter_R(R_wc, yaw_deg=5.0, pitch_deg=3.0)

            cam_perturbation_info = CameraInfo(
                uid=uid, R=R1, T=T, FovY=fovy, FovX=fovx, image=img,
                image_path=image_path, image_name=image_name, width=w, height=h,
                depth=z_depth_np                                                    # === NEW: pass z-depth to downstream
            )
            # cam_perturbation_info.depth = z_depth_np                                 # same z

            cam_perturbation_info_stage2 = CameraInfo(
                uid=uid, R=R2, T=T, FovY=fovy, FovX=fovx, image=img,
                image_path=image_path, image_name=image_name, width=w, height=h,
                depth=z_depth_np                                                    # === NEW: pass z-depth to downstream
            )
            # cam_perturbation_info_stage2.depth = z_depth_np

            cam_perturbation_info_stage3 = CameraInfo(
                uid=uid, R=R3, T=T, FovY=fovy, FovX=fovx, image=img,
                image_path=image_path, image_name=image_name, width=w, height=h,
                depth=z_depth_np                                                    # === NEW: pass z-depth to downstream   
            )
            # cam_perturbation_info_stage3.depth = z_depth_np

            cam_perturbation_infos_unsorted.append(cam_perturbation_info)
            cam_perturbation_infos_unsorted_stage2.append(cam_perturbation_info_stage2)
            cam_perturbation_infos_unsorted_stage3.append(cam_perturbation_info_stage3)

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    cam_perturbation_infos = sorted(cam_perturbation_infos_unsorted.copy(), key=lambda x: x.image_name)
    cam_perturbation_infos_stage2 = sorted(cam_perturbation_infos_unsorted_stage2.copy(), key=lambda x: x.image_name)
    cam_perturbation_infos_stage3 = sorted(cam_perturbation_infos_unsorted_stage3.copy(), key=lambda x: x.image_name)

    llffhold = 8
    if is_eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos  = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos  = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Initialize point cloud directly from ERP ray-length without ad-hoc rescaling
    xyz = pts
    vertex_colors = pano_img.reshape(-1, 3).cpu().numpy()
    ply_path = os.path.join(source_path, 'sparse/0/points3D.ply')
    pcd = BasicPointCloud(points=xyz, colors=vertex_colors, normals=np.zeros_like(xyz, dtype=np.float32))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        perturbation_cameras_stage1=cam_perturbation_infos,
        perturbation_cameras_stage2=cam_perturbation_infos_stage2,
        perturbation_cameras_stage3=cam_perturbation_infos_stage3,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        api_key,
        self_refinement,
        num_prompt,
        max_rounds,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        """
        Core scene initializer.

        Panorama loading priority:
        0) Native Colmap/Blender datasets (unchanged)
        1) Explicit args.pano_path (if provided)
        2) PNG inside args.source_path with filename priority:
           inpainted_panorama.png, inpainted_img.png, panorama.png, diffusion_img.png, image.png
           (fallback: first *.png alphabetically)
        3) Legacy text->panorama generation if only a TXT exists (optionally self-refined)

        After loading a panorama, the image is resized to (2048, 1024),
        geometry is predicted using PanoGeoPredictor, and SceneInfo is built.

        === CHANGED: downstream training now consumes per-view z-depth synthesized
        from ERP ray-length to avoid parallax artifacts and bowl-shaped floors/ceilings.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # Resume if needed
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        # Containers
        self.train_cameras = {}
        self.test_cameras = {}
        self.perturbation_cameras_stage1 = {}
        self.perturbation_cameras_stage2 = {}
        self.perturbation_cameras_stage3 = {}

        # ---------------------------------------------------------------------
        # Priority 0: Colmap/Blender datasets (unchanged)
        # ---------------------------------------------------------------------
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)

        # ---------------------------------------------------------------------
        # Panorama-based path (PNG or TXT present in source_path)
        # ---------------------------------------------------------------------
        elif any(fn.lower().endswith(".png") for fn in os.listdir(args.source_path)) or \
             any(fn.lower().endswith(".txt") for fn in os.listdir(args.source_path)):

            img = None
            img_name = None

            # Priority 1: explicit pano path
            if hasattr(args, "pano_path") and args.pano_path:
                assert os.path.exists(args.pano_path), f"--pano_path not found: {args.pano_path}"
                img_name = args.pano_path
                print(f"[Scene] Using explicit pano_path: {img_name}")
                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            # Priority 2: pick panorama inside source_path by filename priority
            elif any(fn.lower().endswith(".png") for fn in os.listdir(args.source_path)):
                preferred = [
                    "inpainted_panorama.png",
                    "inpainted_img.png",
                    "panorama.png",
                    "diffusion_img.png",
                    "image.png",
                ]
                for cand in preferred:
                    cand_path = os.path.join(args.source_path, cand)
                    if os.path.exists(cand_path):
                        img_name = cand_path
                        print(f"[Scene] Using panorama (priority match): {img_name}")
                        break

                if img_name is None:
                    files = sorted([f for f in os.listdir(args.source_path) if f.lower().endswith(".png")])
                    assert files, "PNG expected but not found"
                    img_name = os.path.join(args.source_path, files[0])
                    print(f"[Scene] Using panorama (fallback first PNG): {img_name}")

                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            # Priority 3: generate from TXT (legacy)
            elif any(fn.lower().endswith(".txt") for fn in os.listdir(args.source_path)):
                sdk = importlib.import_module('stitch_diffusion.kohya_trainer.StitchDiffusionPipeline')
                imgrun = importlib.import_module('Text2PanoRunner')

                txtfile = [f for f in os.listdir(args.source_path) if f.lower().endswith('.txt')][0]
                txt_path = os.path.join(args.source_path, txtfile)

                if self_refinement:
                    assert api_key, "You must enter an api key to access prompt engineered diffusion output"
                    runner = imgrun.Text2PanoRunner(
                        api_key=api_key, testfile=txt_path,
                        num_prompt=num_prompt, max_rounds=max_rounds,
                        foldername=txtfile.rstrip(".txt"),
                    )
                    runner.run_command()
                    best_rel = f"self_refinement/{txtfile.rstrip('.txt')}/iter_best/image.png"
                    assert os.path.exists(best_rel), f"Expected self-refinement output not found: {best_rel}"
                    img_name = os.path.join(args.source_path, "panorama.png")
                    os.system(f"cp '{best_rel}' '{img_name}'")
                    print(f"[Scene] Self-refined panorama copied to: {img_name}")
                else:
                    sd = sdk.StitchDiffusion(sdk.my_args)
                    with open(txt_path, "r") as f:
                        prompt = f.read()
                    out_name = os.path.join(args.source_path, "diffusion_img.png")
                    sd.inference(prompt, savename=out_name)
                    img_name = out_name
                    print(f"[Scene] Basic panorama generated to: {img_name}")

                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            else:
                raise RuntimeError("Could not recognize scene type! (No colmap/blender, no PNG, no TXT)")

            # Canonical resize used downstream
            img = cv.resize(img.cpu().numpy(), (2048, 1024), cv.INTER_AREA)
            img = torch.from_numpy(img).cuda()

            # Geometry prediction from panorama
            geo_predictor = PanoGeoPredictor()
            height, width, _ = img.shape
            distances, rot_w2c, fx, fy, cx, cy, pers_imgs = geo_predictor(img)   # distances is ERP ray-length
            pts = pcd_from_depths(img, distances, height, width, args.source_path)

            print("Saving data for future use...")
            save_data(args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts)

            # === CHANGED: pass args.eval via is_eval to avoid undefined `eval`
            scene_info = get_info_from_params(
                args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts,
                is_eval=bool(getattr(args, "eval", False))
            )

        else:
            raise AssertionError("Could not recognize scene type!")

        # ---------------------------------------------------------------------
        # Boilerplate: export cameras.json and pack camera lists
        # ---------------------------------------------------------------------
        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for cid, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(cid, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
            random.shuffle(scene_info.perturbation_cameras_stage1)
            random.shuffle(scene_info.perturbation_cameras_stage2)
            random.shuffle(scene_info.perturbation_cameras_stage3)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            print("Loading Perturbation Cameras")
            self.perturbation_cameras_stage1[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage1, resolution_scale, args
            )
            self.perturbation_cameras_stage2[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage2, resolution_scale, args
            )
            self.perturbation_cameras_stage3[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage3, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # -------------------------
    # Convenience accessors
    # -------------------------
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPerturbationCameras(self, stage, scale=1.0):
        if stage == 1:
            return self.perturbation_cameras_stage1[scale]
        elif stage == 2:
            return self.perturbation_cameras_stage2[scale]
        elif stage == 3:
            return self.perturbation_cameras_stage3[scale]


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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, SceneInfo ###
from scene.gaussian_model import GaussianModel, BasicPointCloud ###
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, img_coord_to_pano_direction ###
###
from geo_predictors.pano_geo_predictor import *
from utils.utils import read_image
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PIL import Image
import trimesh
import cv2 as cv
from utils.save_data import save_data
import sys
import importlib
sys.path.append('stitch_diffusion/kohya_trainer')

def pcd_from_depths(pano_img, distances, height, width, source_path):
    # ### save pano depth map
    # import matplotlib.pyplot as plt
    # scale_nor = distances.max().item()
    # distances_nor = distances / scale_nor
    # depth_tensor_squeezed = distances_nor.squeeze()  # Remove the channel dimension
    # colormap = plt.get_cmap('jet')
    # depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
    # depth_colored_rgb = depth_colored[:, :, :3]
    # depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
    # output_path = "./pano_depth_map.png"
    # depth_image.save(output_path)
    # ###
    pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
    scale = distances.max().item() * 0.7 #* 0.8 #* 1.05
    distances /= scale
    pts = pano_dirs * distances.squeeze()[..., None]
    pts = pts.cpu().numpy().reshape(-1, 3)
    # pcd = trimesh.PointCloud(pts, pano_img.reshape(-1, 3).cpu().numpy())
    # pcd_path = os.path.join(source_path, 'point_cloud.ply')
    # pcd.export(pcd_path)
    return pts

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def get_info_from_params(source_path, pano_img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts):
        H, W, _ = pano_img.shape
        n_pers, _, h, w = pers_imgs.shape
        cam_infos_unsorted = []
        cam_perturbation_infos_unsorted = [] ###
        cam_perturbation_infos_unsorted_stage2 = [] ###
        cam_perturbation_infos_unsorted_stage3 = [] ###
        for i in range(n_pers):
            with torch.no_grad():
                img = pers_imgs[i].cpu().numpy()
                img = img.transpose(1, 2, 0)
                img = (img*255).astype('uint8')
                img = Image.fromarray(img)
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                fovx = focal2fov(intri['fx'], w)
                fovy = focal2fov(intri['fy'], h)
                R = np.transpose(np.asarray(rot_w2c[i].cpu()))
                T = np.transpose(np.array( [0, 0, 0]))
                T_perturbation = T + np.random.uniform(-0.05, 0.05, size=(1, 3)) ###
                uid = i
                image_name = 'image' + str(i)
                try:
                    os.mkdir( os.path.join ( source_path, 'images'))
                except Exception as e:
                    pass
                image_path = os.path.join(source_path, 'images', image_name)

                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)      
                cam_infos_unsorted.append(cam_info)
                ### stage 1 perturbation
                cam_perturbation_info = CameraInfo(uid=uid, R=R, T=T_perturbation, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted.append(cam_perturbation_info)
                ### stage 2 perturbation
                T_perturbation_stage2 = T + np.random.uniform(-0.05 * 2, 0.05 * 2, size=(1, 3))
                cam_perturbation_info_stage2  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage2, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage2 .append(cam_perturbation_info_stage2)
                ### stage 3 perturbation
                T_perturbation_stage3 = T + np.random.uniform(-0.05 * 4, 0.05 * 4, size=(1, 3))
                cam_perturbation_info_stage3  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage3, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage3 .append(cam_perturbation_info_stage3)



        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        cam_perturbation_infos = sorted(cam_perturbation_infos_unsorted.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage2 = sorted(cam_perturbation_infos_unsorted_stage2.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage3 = sorted(cam_perturbation_infos_unsorted_stage3.copy(), key = lambda x : x.image_name) ###
        llffhold = 8
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            perturbation_cam_infos = cam_perturbation_infos ###
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
            perturbation_cam_infos = cam_perturbation_infos ###
        nerf_normalization = getNerfppNorm(train_cam_infos)
        # #random initialization (comment for using the input pcd)
        # num_pts = 100000
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        xyz = pts #pcd_from_depths(pano_img, distances, H, W, source_path)
        vertex_colors = pano_img.reshape(-1, 3).cpu().numpy()
        ply_path = os.path.join(source_path, 'sparse/0/points3D.ply')
        pcd = BasicPointCloud(points = xyz, colors=vertex_colors, normals=np.zeros_like(xyz))
        scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           perturbation_cameras_stage1=perturbation_cam_infos, ###
                           perturbation_cameras_stage2=cam_perturbation_infos_stage2, ###
                           perturbation_cameras_stage3=cam_perturbation_infos_stage3, ###
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

        return scene_info


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        api_key,
        self_refinement,
        num_prompt,
        max_rounds,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        """
        Core scene initializer.

        Panorama loading priority:
        0) Native Colmap/Blender datasets (unchanged)
        1) Explicit args.pano_path (if provided)
        2) PNG inside args.source_path with filename priority:
           inpainted_panorama.png, inpainted_img.png, panorama.png, diffusion_img.png, image.png
           (fallback: first *.png alphabetically)
        3) Legacy text->panorama generation if only a TXT exists (optionally self-refined)

        After loading a panorama, the image is resized to (2048, 1024),
        geometry is predicted using PanoGeoPredictor, and SceneInfo is built.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # Handle resume from iteration
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        # Containers for camera sets
        self.train_cameras = {}
        self.test_cameras = {}
        self.perturbation_cameras_stage1 = {}
        self.perturbation_cameras_stage2 = {}
        self.perturbation_cameras_stage3 = {}

        # ---------------------------------------------------------------------
        # Priority 0: Colmap/Blender datasets (unchanged)
        # ---------------------------------------------------------------------
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )

        # ---------------------------------------------------------------------
        # Panorama-based path (PNG or TXT present in source_path)
        # ---------------------------------------------------------------------
        elif any(fn.lower().endswith(".png") for fn in os.listdir(args.source_path)) or \
             any(fn.lower().endswith(".txt") for fn in os.listdir(args.source_path)):

            img = None
            img_name = None

            # -------------------------------------------------------------
            # Priority 1: explicit pano path overrides everything else
            # -------------------------------------------------------------
            if hasattr(args, "pano_path") and args.pano_path:
                assert os.path.exists(args.pano_path), f"--pano_path not found: {args.pano_path}"
                img_name = args.pano_path
                print(f"[Scene] Using explicit pano_path: {img_name}")
                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            # -------------------------------------------------------------
            # Priority 2: search inside source_path by filename priority
            # -------------------------------------------------------------
            elif any(fn.lower().endswith(".png") for fn in os.listdir(args.source_path)):
                # NOTE: Keep these names aligned with your pipeline outputs.
                preferred = [
                    "inpainted_panorama.png",  # BG_INPAINT output (preferred)
                    "inpainted_img.png",       # accepted alias if previously used
                    "panorama.png",
                    "diffusion_img.png",
                    "image.png",
                ]
                for cand in preferred:
                    cand_path = os.path.join(args.source_path, cand)
                    if os.path.exists(cand_path):
                        img_name = cand_path
                        print(f"[Scene] Using panorama (priority match): {img_name}")
                        break

                if img_name is None:
                    # Fallback: first *.png alphabetically
                    files = sorted(
                        [f for f in os.listdir(args.source_path) if f.lower().endswith(".png")]
                    )
                    assert files, "PNG expected but not found"
                    img_name = os.path.join(args.source_path, files[0])
                    print(f"[Scene] Using panorama (fallback first PNG): {img_name}")

                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            # -------------------------------------------------------------
            # Priority 3: generate from TXT (legacy, optional self-refine)
            # -------------------------------------------------------------
            elif any(fn.lower().endswith(".txt") for fn in os.listdir(args.source_path)):
                # Import lazily to avoid heavy imports unless needed
                sdk = importlib.import_module('stitch_diffusion.kohya_trainer.StitchDiffusionPipeline')
                imgrun = importlib.import_module('Text2PanoRunner')

                txtfile = [f for f in os.listdir(args.source_path) if f.lower().endswith('.txt')][0]
                txt_path = os.path.join(args.source_path, txtfile)

                if self_refinement:
                    assert api_key, "You must enter an api key to access prompt engineered diffusion output"
                    # Self-refinement via GPT-4o prompt engineering
                    runner = imgrun.Text2PanoRunner(
                        api_key=api_key,
                        testfile=txt_path,
                        num_prompt=num_prompt,
                        max_rounds=max_rounds,
                        foldername=txtfile.rstrip(".txt"),
                    )
                    runner.run_command()
                    best_rel = f"self_refinement/{txtfile.rstrip('.txt')}/iter_best/image.png"
                    assert os.path.exists(best_rel), f"Expected self-refinement output not found: {best_rel}"
                    # Normalize to a canonical file name so downstream always finds it
                    img_name = os.path.join(args.source_path, "panorama.png")
                    os.system(f"cp '{best_rel}' '{img_name}'")
                    print(f"[Scene] Self-refined panorama copied to: {img_name}")
                else:
                    # Basic StitchDiffusion generation
                    sd = sdk.StitchDiffusion(sdk.my_args)
                    with open(txt_path, "r") as f:
                        prompt = f.read()
                    out_name = os.path.join(args.source_path, "diffusion_img.png")
                    sd.inference(prompt, savename=out_name)
                    img_name = out_name
                    print(f"[Scene] Basic panorama generated to: {img_name}")

                img = read_image(img_name, to_torch=True, squeeze=True).cuda()

            else:
                raise RuntimeError("Could not recognize scene type! (No colmap/blender, no PNG, no TXT)")

            # -------------------------------------------------------------
            # Canonical resize used by downstream steps
            # -------------------------------------------------------------
            img = cv.resize(img.cpu().numpy(), (2048, 1024), cv.INTER_AREA)
            img = torch.from_numpy(img).cuda()

            # -------------------------------------------------------------
            # Geometry prediction & SceneInfo construction (unchanged)
            # -------------------------------------------------------------
            geo_predictor = PanoGeoPredictor()
            height, width, _ = img.shape
            distances, rot_w2c, fx, fy, cx, cy, pers_imgs = geo_predictor(img)
            pts = pcd_from_depths(img, distances, height, width, args.source_path)

            print("Saving data for future use...")
            save_data(
                args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts
            )
            scene_info = get_info_from_params(
                args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts
            )

        else:
            raise AssertionError("Could not recognize scene type!")

        # ---------------------------------------------------------------------
        # Boilerplate below remains identical
        # ---------------------------------------------------------------------
        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for cid, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(cid, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
            random.shuffle(scene_info.perturbation_cameras_stage1)
            random.shuffle(scene_info.perturbation_cameras_stage2)
            random.shuffle(scene_info.perturbation_cameras_stage3)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            print("Loading Perturbation Cameras")
            self.perturbation_cameras_stage1[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage1, resolution_scale, args
            )
            self.perturbation_cameras_stage2[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage2, resolution_scale, args
            )
            self.perturbation_cameras_stage3[resolution_scale] = cameraList_from_camInfos(
                scene_info.perturbation_cameras_stage3, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # -------------------------
    # Convenience accessors
    # -------------------------
    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPerturbationCameras(self, stage, scale=1.0):
        if stage == 1:
            return self.perturbation_cameras_stage1[scale]
        elif stage == 2:
            return self.perturbation_cameras_stage2[scale]
        elif stage == 3:
            return self.perturbation_cameras_stage3[scale]

'''