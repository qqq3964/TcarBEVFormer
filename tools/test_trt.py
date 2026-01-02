#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-GPU TensorRT evaluation script for BEVFormer_TensorRT
- Loads TensorRT engine (.plan / .engine)
- Runs inference frame by frame
- Reconstructs prev_bev logic
- Calls post_process from the PyTorch model definition to get bbox results
- Computes NuScenes metrics via dataset.evaluate()

Usage (through the wrapper script we made):
    CUDA_VISIBLE_DEVICES=0 python test_trt.py configs/xxx.py path/to/model.engine
"""

import argparse
import os
import sys
import copy
import time
import numpy as np
import torch
import mmcv
from mmcv import Config
# from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import ctypes
import pycuda.autoinit          # NOTE: assumes CUDA_VISIBLE_DEVICES already set by caller
import pycuda.driver as cuda
import tensorrt as trt

# make sure local project modules are visible
sys.path.append(".")

from tensorRT.utils import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)

# these builders/dataloaders are from your TensorRT fork (third_party/...)
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset


#from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BEVFormer TensorRT engine on val/test split"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument(
        "trt_model",
        help="TensorRT engine file (.engine / .plan). This replaces the PyTorch checkpoint",
    )
    # you could add --workers, --samples-per-gpu 등 필요한 옵션 나중에 확장 가능
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. TensorRT plugin 로드 (custom BEVFormer ops 등)
    # load_tensorrt_plugin()

    # --- .so 파일을 ctypes로 직접 로드 ---
    plugin_path = "/home/user/BEVFormer/tools/tensorRT/libmmdeploy_tensorrt_ops.so"
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"Custom TRT plugin not found: {plugin_path}\n"
            "Make sure you have compiled the mmdeploy plugins and the path is correct."
        )
    try:
        # CDLL을 호출하는 것만으로도 라이브러리가 프로세스 메모리에 로드됩니다.
        ctypes.CDLL(plugin_path)
        print(f"Successfully loaded custom plugin: {plugin_path}")
    except OSError as e:
        print(f"Error loading plugin {plugin_path}: {e}")
        print("Please check library dependencies (e.g., using ldd)")
        sys.exit(1)

    trt_engine_path = args.trt_model
    config_file = args.config

    # TensorRT logger / context 준비
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)
    engine, context = create_engine_context(trt_engine_path, TRT_LOGGER)
    stream = cuda.Stream()

    # 2. config 로드 & plugin import (cfg.plugin 지원)
    cfg = Config.fromfile(config_file)

    if hasattr(cfg, "plugin"):
        import importlib

        # cfg.plugin 이 list일 수도 있고 string일 수도 있음
        if isinstance(cfg.plugin, list):
            for plu in cfg.plugin:
                importlib.import_module(plu)
        elif isinstance(cfg.plugin, str):
                importlib.import_module(cfg.plugin)

    # TensorRT I/O shape 정보
    output_shapes_cfg = cfg.output_shapes      # dict: name -> shape list w/ strings
    input_shapes_cfg = cfg.input_shapes        # dict: name -> shape list w/ strings
    default_shapes = cfg.default_shapes        # dict of symbols like B, H, W, etc.

    # 예: cfg.default_shapes = {"B": 1, "H": 900, ...}
    # 이런 것들을 현재 로컬 네임스페이스 변수로 풀어준다 (eval()에서 쓰려고)
    for key in default_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined already.")
        locals()[key] = default_shapes[key]

    samples_per_gpu_val = cfg.data.val.pop('samples_per_gpu', 1)
    # 3. dataset / dataloader 구성
    #   - 원본 evaluate_trt.py는 cfg.data.val 을 사용함
    #   - 너희 일반 eval 코드(test.py 스타일)는 cfg.data.test 를 사용함
    #   여기서는 TensorRT eval 관례 그대로 따라가서 cfg.data.val 사용
    dataset = build_dataset(cfg=cfg.data.val)
    
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu_val,
        workers_per_gpu=getattr(cfg.data, "workers_per_gpu", 6),
        shuffle=False,
        dist=False,
    )

    # 4. 후처리를 위해 PyTorch model graph 빌드 (weights는 안 쓸거고 post_process만 씀)
    pth_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    bbox_results = []
    times_ms = []
    prog_bar = mmcv.ProgressBar(len(dataset))

    # BEVFormer temporal memory
    # prev_bev: (bev_h * bev_w, B, dim)
    prev_bev = np.random.randn(cfg.bev_h_ * cfg.bev_w_, 1, cfg._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    # 5. 메인 inference 루프
    for data in loader:
        # data 구조: same as mmdet style
        # img: list[DataContainer(Tensor)] etc.
        img = data["img"][0].data[0].numpy()          # shape: (B, Ncams, C, H, W)
        img_metas = data["img_metas"][0].data[0]      # list of dicts length B

        # temporal BEV reuse 여부
        use_prev_bev = np.array([1.0], dtype=np.float32)
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0], dtype=np.float32)

        # track scene info
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]

        # copy pose info before we overwrite can_bus
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])

        # relative motion encoding
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0

        can_bus = img_metas[0]["can_bus"].astype(np.float32)
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0).astype(np.float32)

        batch_size, num_cams, _, img_h, img_w = img.shape  # B, Ncams, C, H, W

        # 5-1. output / input shape eval해서 실제 텐서 크기 풀기
        # (cfg에서 문자열("B","H","W"...)로 들어있는 걸 실제 숫자로 대체)
        resolved_out_shapes = {}
        for name, shape_spec in output_shapes_cfg.items():
            shape_list = []
            for dim_val in shape_spec:
                if isinstance(dim_val, str):
                    shape_list.append(eval(dim_val))
                else:
                    shape_list.append(dim_val)
            resolved_out_shapes[name] = shape_list

        resolved_in_shapes = {}
        for name, shape_spec in input_shapes_cfg.items():
            shape_list = []
            for dim_val in shape_spec:
                if isinstance(dim_val, str):
                    shape_list.append(eval(dim_val))
                else:
                    shape_list.append(dim_val)
            resolved_in_shapes[name] = shape_list

        # 5-2. TensorRT 입출력 버퍼 준비
        inputs, outputs, bindings = allocate_buffers(
            engine,
            context,
            input_shapes=resolved_in_shapes,
            output_shapes=resolved_out_shapes,
        )

        # 5-3. host 메모리에 실제 데이터 채우기
        for buf in inputs:
            if buf.name == "image":
                # expect float32 flattened
                buf.host = img.reshape(-1).astype(np.float32)
            elif buf.name == "prev_bev":
                buf.host = prev_bev.reshape(-1).astype(np.float32)
            elif buf.name == "use_prev_bev":
                buf.host = use_prev_bev.reshape(-1).astype(np.float32)
            elif buf.name == "can_bus":
                buf.host = can_bus.reshape(-1).astype(np.float32)
            elif buf.name == "lidar2img":
                buf.host = lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected TRT input name {buf.name}")

        # 5-4. Inference
        trt_raw_outputs, t_ms = do_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        # t_ms from do_inference() in upstream code is usually seconds, we convert later

        # 5-5. output reshape & temporal state 업데이트
        trt_outputs = {}
        for out in trt_raw_outputs:
            trt_outputs[out.name] = out.host.reshape(*resolved_out_shapes[out.name])

        # keep BEV memory for next frame
        prev_bev = trt_outputs.pop("bev_embed")

        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        # convert numpy -> torch for post_process
        trt_outputs_torch = {
            k: torch.from_numpy(v) for k, v in trt_outputs.items()
        }

        # 5-6. decode boxes using PyTorch model's post_process
        # pth_model.post_process(...) should return list[dict/list] per sample
        bbox_results.extend(
            pth_model.post_process(**trt_outputs_torch, img_metas=img_metas)
        )

        # log time
        times_ms.append(t_ms)

        # progress bar
        for _ in range(len(img)):
            prog_bar.update()

    # 6. metric 계산
    metric = dataset.evaluate(bbox_results)

    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        # 이 이름들은 NuScenes 평가 결과에서 자주 나오는 key
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")

    # latency / fps 계산 (처음/마지막 프레임 버리고 평균내는 패턴 유지)
    if len(times_ms) > 2:
        # do_inference()의 t_ms 가 "seconds" 단위로 나오는 구현도 있어서 방어적으로 처리
        # 만약 t_ms 가 이미 ms 단위라면 아래 / *1000 부분 조정 필요.
        # 여기서는 upstream 코드 로직 따라:
        # latency(ms) = avg(중간 step들) * 1000
        latency_ms = round(sum(times_ms[1:-1]) / len(times_ms[1:-1]) * 1000, 2)
    else:
        latency_ms = round(sum(times_ms) / max(len(times_ms), 1) * 1000, 2)

    print(f"Latency: {latency_ms}ms")
    if latency_ms > 0:
        print(f"FPS: {1000.0 / latency_ms}")
    else:
        print("FPS: inf")


if __name__ == "__main__":
    main()
