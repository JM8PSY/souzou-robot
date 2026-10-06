"""テスト用の合成シーン（天井カメラから見たフィールドとマーカー）を描画する．"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from localization.camera_model import CameraModel

IMAGE_SIZE = (1280, 720)
FIELD_W, FIELD_H = 1760.0, 900.0
CAMERA_HEIGHT = 2500.0
F_TRUE = 900.0


def make_camera(tilt_deg: float = 5.0, cam_xy=(880.0, 450.0), height=CAMERA_HEIGHT) -> CameraModel:
    """真下向きから tilt_deg だけカメラ x 軸回りに傾けた天井カメラ．"""
    # world -> camera：カメラ x = 世界 +X，カメラ y = 世界 -Y，カメラ z = 世界 -Z（真下）
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    a = math.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    R = Rx @ R0
    C = np.array([cam_xy[0], cam_xy[1], height])
    t = -R @ C
    rvec, _ = cv2.Rodrigues(R)
    w, h = IMAGE_SIZE
    K = np.array([[F_TRUE, 0, w / 2], [0, F_TRUE, h / 2], [0, 0, 1]])
    return CameraModel(K, np.zeros(5), rvec, t)


@dataclass
class SceneMarker:
    id: int
    x: float
    y: float
    theta: float  # マーカー上方向（ロボット前方）の向き [rad]
    height_mm: float
    size_mm: float


def marker_world_corners(m: SceneMarker) -> np.ndarray:
    """マーカーの黒い部分の 4 隅（ArUco の順: 左上, 右上, 右下, 左下）を世界座標で返す．"""
    up = np.array([math.cos(m.theta), math.sin(m.theta)])
    right = np.array([math.sin(m.theta), -math.cos(m.theta)])
    c = np.array([m.x, m.y])
    s = m.size_mm / 2
    pts = [c - right * s + up * s, c + right * s + up * s, c + right * s - up * s, c - right * s - up * s]
    return np.array([[p[0], p[1], m.height_mm] for p in pts])


def render(camera: CameraModel, markers: list[SceneMarker], dictionary: str = "DICT_4X4_50",
           dark_border_px: int = 0) -> np.ndarray:
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), 255, np.uint8)  # 白い壁
    if dark_border_px:
        # 部屋の床など，画像端に接する暗い領域
        img[:dark_border_px, :] = 30
        img[-dark_border_px:, :] = 30
        img[:, :dark_border_px] = 30
        img[:, -dark_border_px:] = 30
    floor = camera.field_to_pixel(
        [[0, 0, 0], [FIELD_W, 0, 0], [FIELD_W, FIELD_H, 0], [0, FIELD_H, 0]]
    )
    cv2.fillConvexPoly(img, np.round(floor).astype(np.int32), (20, 20, 20))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    for m in markers:
        side = 120
        pad = side // 6  # 白いクワイエットゾーン（1 セル分）
        tag = cv2.aruco.generateImageMarker(d, m.id, side)
        tag = cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        total = side + 2 * pad
        padded = SceneMarker(m.id, m.x, m.y, m.theta, m.height_mm, m.size_mm * total / side)
        dst = camera.field_to_pixel(marker_world_corners(padded)).astype(np.float32)
        src = np.array([[0, 0], [total, 0], [total, total], [0, total]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tag, H, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((total, total), 255, np.uint8), H, (w, h))
        img[mask > 127] = warped[mask > 127]
    return img
