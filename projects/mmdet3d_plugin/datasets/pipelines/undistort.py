import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import cv2
import time

INTRINSICS = [None] * 6
MAPS1 = [None] * 6
MAPS2 = [None] * 6

@PIPELINES.register_module()
class MyCustomUndistort:
    def __init__(self, balance=0.0):
        """Fisheye undistortion pipeline.
        Args:
            balance (float): 0.0 for max crop, 1.0 for full FOV.
        """
        self.balance = balance

    def __call__(self, results):
        imgs = results['img']
        if 'img_shape' in results:
            h, w = results['img_shape'][:2]
        else:
            h, w = imgs[0].shape[:2]

        undistorted_imgs = []
        for idx, img in enumerate(imgs):
            start = time.time()
            undistorted_imgs.append(self.undistort(img, h, w, results, idx))
            print(f"[Cam {idx}] undistort time: {(time.time() - start)*1000:.2f} ms")
            
            assert INTRINSICS[idx] is not None
            results['cam_intrinsic'][idx] = INTRINSICS[idx]
            results['lidar2img'][idx] = INTRINSICS[idx] @ results['lidar2cam'][idx]

        results['img'] = undistorted_imgs
        return results

    def undistort(self, img, h, w, results, idx):
        """Perform fisheye or pinhole undistortion per camera."""
        
        dist_intrinsic = results['cam_intrinsic'][idx]
        distortion = np.array(results['distortion'][idx], dtype=np.float64).flatten()
        
        if dist_intrinsic.shape == (4, 4):
            dist_intrinsic = dist_intrinsic[:3, :3]
            
        if idx != 0 and INTRINSICS[idx] is None: # not cam front
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                dist_intrinsic, distortion, (w, h), np.eye(3), balance=self.balance
            )
            MAPS1[idx], MAPS2[idx] = cv2.fisheye.initUndistortRectifyMap(
                dist_intrinsic, distortion, np.eye(3),
                new_K, (w, h), cv2.CV_16SC2
            )
            INTRINSICS[idx] = np.eye(4, dtype=np.float64)
            INTRINSICS[idx][:3, :3] = new_K
        elif idx == 0 and INTRINSICS[idx] is None: # cam front
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                dist_intrinsic, distortion, (w, h), 0
            )
            INTRINSICS[idx] = np.eye(4, dtype=np.float64)
            INTRINSICS[idx][:3, :3] = new_K
        
        if idx != 0:
            img = cv2.remap(img, MAPS1[idx], MAPS2[idx], interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.undistort(img, dist_intrinsic, distortion, None, INTRINSICS[idx][:3, :3])

        return img