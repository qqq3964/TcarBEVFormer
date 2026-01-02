# -*- coding: utf-8 -*-
"""
Composite scene visualizer:
- Left: 3x2 camera mosaic with predicted boxes projected on each image (no gaps).
- Right: BEV view (LiDAR XY scatter + GT/PRED boxes) sized to exactly match the LEFT block height,
         auto-scaled to fully fill the right column width, and rotated 90° CCW.
- Saves one PNG per sample (frame), then stitches frames into an MP4.

Notes:
- All comments are in English as requested.
- Requires: nuscenes-devkit, mmcv, matplotlib, pillow, pyquaternion, opencv-python (optional, for mp4), tqdm (optional).
- Assumes BEVFormer-style results_nusc.json structure under pred_data['results'][sample_token].
"""

import os
import sys
import os.path as osp
import glob
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image
from matplotlib.gridspec import GridSpec
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

# Project wrapper (adjust import path if needed)
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from tools.tcar.tcar import TestCar
from tools.tcar.utils import LidarPointCloud

# Deterministic rendering rcParams
mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['figure.constrained_layout.use'] = False


# ------------------------------
# Utility: color lookup by category name
# ------------------------------
def get_color(nusc, category_name: str):
    """
    Map a (pred) category name to the NuScenes colormap color.
    Fallback to black if not found.
    """
    name_map = {
        'bicycle': 'vehicle.bicycle',
        'construction_vehicle': 'vehicle.construction',
        'traffic_cone': 'movable_object.trafficcone'
    }
    key = name_map.get(category_name, None)
    if key and key in nusc.colormap:
        return nusc.colormap[key]
    # Try substring match
    for k in nusc.colormap.keys():
        if category_name in k:
            return nusc.colormap[k]
    return [0, 0, 0]


# --- Robust accessors for NuScenes Box/EvalBox variants ---
def _extract_center_size_quat_name(b):
    """
    Extract (center, size, quaternion, name) from either:
    - nuscenes.utils.data_classes.Box (center, size or wlh, orientation, name)
    - EvalBox-like objects (translation, size, rotation, detection_name)
    """
    if hasattr(b, 'center'):
        center = np.array(b.center)
    elif hasattr(b, 'translation'):
        center = np.array(b.translation)
    else:
        raise AttributeError("Box-like object has neither 'center' nor 'translation'.")

    if hasattr(b, 'size'):
        size = np.array(b.size)
    elif hasattr(b, 'wlh'):
        size = np.array(b.wlh)
    else:
        raise AttributeError("Box-like object has neither 'size' nor 'wlh'.")

    if hasattr(b, 'orientation'):
        quat = b.orientation
    elif hasattr(b, 'rotation'):
        quat = Quaternion(b.rotation)
    else:
        raise AttributeError("Box-like object has neither 'orientation' nor 'rotation'.")

    name = getattr(b, 'name', getattr(b, 'detection_name', 'vehicle'))
    return center, size, quat, name


# ------------------------------
# Project predicted boxes into a given camera
# ------------------------------
def get_predicted_data(nusc,
                       sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       pred_anns=None):
    """
    Returns (data_path, projected_boxes, camera_intrinsic) for a camera sample_data.
    `pred_anns` must be a list of nuscenes Box objects in EGO frame.
    We transform them into the target camera sensor frame and keep the ones visible in the image.
    """
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] != 'camera':
        raise ValueError("get_predicted_data() expects a camera sample_data_token.")

    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # Use real image size for visibility filtering
    try:
        _im = Image.open(data_path)
        imsize = _im.size  # (width, height)
        _im.close()
    except Exception:
        imsize = (1920, 1080)

    box_list = []
    for box in pred_anns:
        c, s, q, nm = _extract_center_size_quat_name(box)
        b = Box(c.copy(), s.copy(), q, name=nm)

        # ego -> sensor(camera)
        b.rotate(Quaternion(cs_record['rotation']))
        b.translate(np.array(cs_record['translation']))

        if not box_in_image(b, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(b)

    return data_path, box_list, cam_intrinsic


# ------------------------------
# Helper: get GT boxes in LiDAR frame via NuScenes API
# ------------------------------
def get_gt_boxes_in_lidar(nusc, sample_token: str, lidar_sd_token: str):
    """
    Return GT boxes already transformed into the LiDAR sensor frame
    using the official NuScenes helper (most reliable).
    """
    sample = nusc.get('sample', sample_token)
    _, boxes_lidar, _ = nusc.get_sample_data(
        lidar_sd_token,
        selected_anntokens=sample['anns']
    )
    return boxes_lidar


# ------------------------------
# Render one composite frame (left 3x2 cams, right BEV matched to left height)
# ------------------------------
def render_composite_frame(nusc,
                           sample_token: str,
                           pred_data: dict,
                           out_path: str,
                           cams_order=('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'),
                           score_th: float = 0.5,
                           dpi: int = 150,
                           bev_range: float = 50.0):
    """
    Build a single composite frame and save to `out_path` (PNG).

    - Left: 3x2 tightly packed camera images with predicted boxes (score >= score_th).
    - Right: BEV axes HEIGHT exactly equals LEFT block height, width fills its column;
             auto-scaled to the pixel aspect of the right column and rotated 90° CCW.
    """
    sample = nusc.get('sample', sample_token)

    # --- Probe one camera image to get native size ---
    probe_sd = sample['data'][cams_order[0]]
    probe_path = nusc.get_sample_data_path(probe_sd)
    with Image.open(probe_path) as _im:
        img_w, img_h = _im.size  # e.g., 1600x900

    # --- Set figure size in exact pixels so tiles == image size ---
    fig_w_px = 4 * img_w
    fig_h_px = 2 * img_h
    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    # Solid black background for the entire figure
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(1.0)

    # Zero gaps
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    gs_left = gs[0, 0].subgridspec(2, 3, wspace=0.0, hspace=0.0)
    left_axes = [fig.add_subplot(gs_left[i // 3, i % 3]) for i in range(6)]
    for ax in left_axes:
        ax.set_facecolor('black')
        ax.axis('off')

    # Build Box list in EGO frame from predictions for this sample
    pred_records = pred_data['results'].get(sample_token, [])
    pred_records = sorted(pred_records, key=lambda r: (-r.get('detection_score', 0.0),
                                                       r.get('sample_token', ''),
                                                       r.get('detection_name', '')))
    pred_boxes_ego = []
    for rec in pred_records:
        if rec.get('detection_score', 0.0) < score_th:
            continue
        pred_boxes_ego.append(
            Box(center=np.array(rec['translation']),
                size=np.array(rec['size']),
                orientation=Quaternion(rec['rotation']),
                name=rec.get('detection_name', 'vehicle'))
        )

    # Render each camera tile with projected predictions
    for idx, cam in enumerate(cams_order):
        ax = left_axes[idx]
        sd_token = sample['data'][cam]
        data_path, boxes_proj, K = get_predicted_data(nusc,
                                                      sd_token,
                                                      box_vis_level=BoxVisibility.ANY,
                                                      pred_anns=pred_boxes_ego)
        img = Image.open(data_path)
        w, h = img.size
        ax.imshow(img, interpolation='none')  # avoid anti-aliased seams
        for b in boxes_proj:
            col = np.array(get_color(nusc, b.name)) / 255.0
            b.render(ax, view=K, normalize=True, colors=(col, col, col), linewidth=1)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.axis('off')

    # --- Compute positions of left and right columns in figure fraction ---
    left_holder = fig.add_subplot(gs[0, 0]); left_holder.axis('off')
    right_holder = fig.add_subplot(gs[0, 1]); right_holder.axis('off')
    fig.canvas.draw()
    left_pos, right_pos = left_holder.get_position(), right_holder.get_position()
    left_holder.remove(); right_holder.remove()

    # --- Right panel: BEV axes (height = left block height) ---
    bev_ax = fig.add_axes([right_pos.x0, left_pos.y0, right_pos.width, left_pos.height])
    bev_ax.set_facecolor('black')
    bev_ax.axis('off')
    bev_ax.set_aspect('equal', adjustable='box')

    # ------------------------------
    # Load LiDAR point cloud and draw as XY scatter (rotated 90° CCW)
    # ------------------------------
    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
    lidar_path = osp.join(nusc.dataroot, sd_lidar['filename'])
    pc = LidarPointCloud.from_file(lidar_path)

    # Rotate 90° CCW: (x, y) -> (-y, x)
    xy = pc.points[:2, :]  # (2, N)
    x_rot = -xy[1, :]
    y_rot =  xy[0, :]

    # Colorize points by radial distance (normalized to vertical range)
    dists = np.sqrt(x_rot ** 2 + y_rot ** 2)
    # We'll normalize by the vertical radius (bev_range); horizontal will be computed to fill the width.
    colors = np.clip(dists / float(bev_range), 0.25, 1.0)
    # Before setting limits, compute pixel aspect of the BEV axes to auto-scale X to fill width
    fig.canvas.draw()
    bbox_px = bev_ax.get_window_extent(fig.canvas.get_renderer())
    px_aspect = bbox_px.width / bbox_px.height  # (pixels_x / pixels_y)

    # Vertical limits are fixed to [-bev_range, bev_range] in meters (rotated Y axis)
    ymin, ymax = -bev_range, bev_range
    yr = ymax - ymin
    xr = yr * px_aspect  # choose X span so that meters-per-pixel is isotropic and width is fully filled
    xmin, xmax = -xr / 2.0, xr / 2.0

    # Scatter with small dot marker and no edge to avoid grid-like artifacts
    bev_ax.scatter(x_rot, y_rot, c=colors, s=0.9, alpha=0.9, cmap='viridis',
                   marker='.', linewidths=0, rasterized=True)
    # Ego mark
    bev_ax.plot(0, 0, 'x', color='white')

    # ------------------------------
    # Render GT and Pred boxes in LiDAR frame, with 90° CCW view matrix
    # ------------------------------
    # Prediction boxes (ego -> lidar)
    boxes_pred_lidar = []
    for rec in pred_records:
        if rec.get('detection_score', 0.0) < score_th:
            continue
        b = Box(center=np.array(rec['translation']),
                size=np.array(rec['size']),
                orientation=Quaternion(rec['rotation']),
                name=rec.get('detection_name', 'vehicle'))
        b.rotate(Quaternion(cs_lidar['rotation']))
        b.translate(np.array(cs_lidar['translation']))
        boxes_pred_lidar.append(b)

    # Ground-truth boxes (already in lidar frame from API)
    boxes_gt_lidar = get_gt_boxes_in_lidar(nusc, sample_token, sample['data']['LIDAR_TOP'])

    # 90° CCW rotation as a view matrix for box rendering
    R_ccw90 = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ], dtype=float)

    pred_color = (0.0, 0.85, 1.0)  # cyan
    gt_color   = (1.0, 0.0, 0.0)   # red

    for box in boxes_gt_lidar:
        box.render(bev_ax, view=R_ccw90, colors=(gt_color, gt_color, gt_color), linewidth=2.2)
    for box in boxes_pred_lidar:
        box.render(bev_ax, view=R_ccw90, colors=(pred_color, pred_color, pred_color), linewidth=1.8)

    # Apply the auto-scaled limits (fits the right column exactly with equal aspect in meters)
    bev_ax.set_xlim(xmin, xmax)
    bev_ax.set_ylim(ymin, ymax)
    bev_ax.set_aspect('equal')

    # Hide ticks / grid / spines completely
    bev_ax.set_xticks([])
    bev_ax.set_yticks([])
    bev_ax.grid(False)
    for spine in bev_ax.spines.values():
        spine.set_visible(False)
    bev_ax.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=pred_color, lw=2, label='Pred'),
        Line2D([0], [0], color=gt_color,   lw=2, label='GT'),
    ]
    bev_ax.legend(handles=legend_elems, loc='upper right', frameon=False,
                  fontsize=20, labelcolor='white')

    # Save PNG frame (no tight bbox -> consistent canvas)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------
# Optional: build mp4 via OpenCV (if you don't want to use ffmpeg)
# ------------------------------
def build_video_from_frames_glob(frame_glob: str, out_mp4: str, fps: int = 10):
    """
    Stitch PNG frames into an MP4 using OpenCV. Use when ffmpeg is not preferred.
    - frame_glob example: 'vis_out/scene_010/*.png'
    """
    import cv2
    frames = sorted(glob.glob(frame_glob))
    assert frames, f"No frames match: {frame_glob}"
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(img)
    vw.release()
    print(f"[OK] Wrote video: {out_mp4}")


# ------------------------------
# Main: render all samples in a chosen scene to frames, then make video
# ------------------------------
if __name__ == '__main__':
    """
    How to use:
    1) Adjust `version`, `dataroot`, `results_json`, and `scene_idx` as needed.
    2) Run:  python3 path/to/this_script.py
    3) Frames will be saved under vis_out/<scene_name>/00000.png ...
    4) Make MP4:
       - (A) ffmpeg (recommended):
           sudo apt-get update && sudo apt-get install -y ffmpeg
           ffmpeg -y -framerate 10 -i vis_out/n030/%05d.png \
             -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
             vis_out/n030_composite.mp4
       - (B) OpenCV:
           build_video_from_frames_glob('vis_out/n030/*.png', 'vis_out/n030_composite.mp4', fps=10)
    """

    # -------------- Config --------------
    version = 'v1.0-trainval'
    dataroot = './data/nuscenes'
    results_json = 'test/bevformer_tiny_tcar/Fri_Oct_10_22_17_15_2025/pts_bbox/results_nusc.json'
    scene_idx = 16
    cams_order = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                  'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')
    score_th = 0.5
    dpi = 150
    bev_range = 50.0
    fps = 5
    # ------------------------------------

    # Init dataset
    nusc = TestCar(version=version, dataroot=dataroot, verbose=True)

    # Load predictions
    pred = mmcv.load(results_json)

    # Resolve scene name from your split helper
    from tools.tcar import splits
    val_scenes = splits.val
    assert 0 <= scene_idx < len(val_scenes), f"scene_idx out of range: {scene_idx}"
    scene_name = val_scenes[scene_idx]

    # Find exact scene record
    scene_record = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene_record = s
            break
    assert scene_record is not None, f"Scene not found: {scene_name}"

    # Collect sample tokens in this scene
    sample_tokens = []
    tok = scene_record['first_sample_token']
    while tok:
        sample_tokens.append(tok)
        sample = nusc.get('sample', tok)
        tok = sample['next']

    print(f"[INFO] Scene index: {scene_idx}  ({scene_record['name']})")
    print(f"[INFO] Number of samples: {len(sample_tokens)}")

    # Prepare output directory
    out_dir = osp.join('vis_out', scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # Render frames
    for i, st in enumerate(sample_tokens):
        out_png = osp.join(out_dir, f"{i:05d}.png")
        render_composite_frame(nusc,
                               st,
                               pred_data=pred,
                               out_path=out_png,
                               cams_order=cams_order,
                               score_th=score_th,
                               dpi=dpi,
                               bev_range=bev_range)
        print(f"[OK] Saved: {out_png}")

    print("[DONE] All frames saved.")

    # Optionally stitch to MP4 here using OpenCV:
    build_video_from_frames_glob(osp.join(out_dir, "*.png"),
                                 osp.join(out_dir, f"{scene_name}_composite.mp4"),
                                 fps=fps)
