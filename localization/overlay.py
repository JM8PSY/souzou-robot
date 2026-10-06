"""カメラ映像への重畳描画と，フィールドの俯瞰図．"""
import math

import cv2
import numpy as np

from localization.camera_model import CameraModel
from localization.config import Config
from localization.pose import STATUS_OK, PoseResult

GREEN, YELLOW, RED, CYAN, BLUE, WHITE = (0, 200, 0), (0, 220, 220), (0, 0, 255), (255, 255, 0), (255, 120, 0), (255, 255, 255)
STATUS_COLOR = {"ok": GREEN, "estimated": YELLOW, "lost": RED}


def _ipt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_camera_view(img_bgr: np.ndarray, camera: CameraModel, cfg: Config, corners_px: np.ndarray,
                     detections: dict[int, np.ndarray], result: PoseResult) -> np.ndarray:
    out = img_bgr.copy()
    cv2.polylines(out, [np.round(corners_px).astype(np.int32)], True, CYAN, 2)
    # 原点と軸（長さ 200mm）
    o, ax, ay = camera.field_to_pixel([[0, 0, 0], [200, 0, 0], [0, 200, 0]])
    cv2.arrowedLine(out, _ipt(o), _ipt(ax), RED, 2)
    cv2.arrowedLine(out, _ipt(o), _ipt(ay), GREEN, 2)
    cv2.putText(out, "O", _ipt(o), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    # 中央線（越えると反則）と自陣側の印．向きの設定ミスに気づけるようにする
    (x1, y1), (x2, y2) = cfg.center_line
    l1, l2, own = camera.field_to_pixel([[x1, y1, 0], [x2, y2, 0], [*cfg.own_side_point, 0]])
    cv2.line(out, _ipt(l1), _ipt(l2), RED, 3)
    cv2.putText(out, "OWN", _ipt(own), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    for mid, c in detections.items():
        cv2.polylines(out, [np.round(c).astype(np.int32)], True, BLUE, 2)
        cv2.putText(out, str(mid), _ipt(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
    color = STATUS_COLOR.get(result.status, RED)
    if result.pose is not None:
        p = result.pose
        h = cfg.center.height_mm
        a, b = camera.field_to_pixel([[p.x, p.y, h],
                                      [p.x + 150 * math.cos(p.theta), p.y + 150 * math.sin(p.theta), h]])
        cv2.arrowedLine(out, _ipt(a), _ipt(b), color, 3)
    for arm, spec in ((result.arm_left, cfg.left), (result.arm_right, cfg.right)):
        if arm is not None:
            q = camera.field_to_pixel([[arm.x, arm.y, spec.height_mm]])[0]
            cv2.circle(out, _ipt(q), 8, color, 2)
    text = f"{result.status}"
    if result.pose is not None:
        text += f"  x={result.pose.x:.0f} y={result.pose.y:.0f} th={math.degrees(result.pose.theta):.1f}deg"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if result.warnings:
        cv2.putText(out, "WARNING: " + ",".join(result.warnings), (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    return out


def draw_top_view(result: PoseResult, cfg: Config, px_per_mm: float = 0.4, margin: int = 20) -> np.ndarray:
    """フィールド座標の俯瞰図（+Y が画面上向き）．"""
    w = int(cfg.field_width_mm * px_per_mm) + 2 * margin
    h = int(cfg.field_height_mm * px_per_mm) + 2 * margin
    out = np.full((h, w, 3), 40, np.uint8)

    def to_px(x, y):
        return int(round(margin + x * px_per_mm)), int(round(h - margin - y * px_per_mm))

    cv2.rectangle(out, to_px(0, 0), to_px(cfg.field_width_mm, cfg.field_height_mm), WHITE, 2)
    (x1, y1), (x2, y2) = cfg.center_line
    cv2.line(out, to_px(x1, y1), to_px(x2, y2), RED, 2)
    color = STATUS_COLOR.get(result.status, RED)
    if result.pose is not None:
        p = result.pose
        cv2.circle(out, to_px(p.x, p.y), 6, color, -1)
        cv2.arrowedLine(out, to_px(p.x, p.y),
                        to_px(p.x + 150 * math.cos(p.theta), p.y + 150 * math.sin(p.theta)), color, 2)
    for arm in (result.arm_left, result.arm_right):
        if arm is not None:
            cv2.circle(out, to_px(arm.x, arm.y), 5, CYAN, -1)
            if result.pose is not None:
                cv2.line(out, to_px(result.pose.x, result.pose.y), to_px(arm.x, arm.y), CYAN, 1)
    if result.warnings:
        cv2.putText(out, "WARNING: " + ",".join(result.warnings), (margin, margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return out
