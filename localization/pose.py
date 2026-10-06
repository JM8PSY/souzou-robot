"""検出したマーカーからロボット姿勢とアーム先端位置を求める．"""
import math
from dataclasses import dataclass

import numpy as np

from localization.camera_model import CameraModel
from localization.config import Config, MarkerSpec

STATUS_OK = "ok"
STATUS_ESTIMATED = "estimated"
STATUS_LOST = "lost"
WARN_CROSS_CENTER = "cross_center_line"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float  # +X 軸から反時計回り [rad]，(-pi, pi]


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class PoseResult:
    t: float
    status: str
    lost_for_s: float
    pose: Pose2D | None
    arm_left: Point2D | None
    arm_right: Point2D | None
    warnings: tuple[str, ...]
    last_seen_t: float | None  # 最後に姿勢が得られた時刻（出力 JSON には含めない）

    def to_dict(self) -> dict:
        def pt(p):
            return None if p is None else {"x": round(p.x, 1), "y": round(p.y, 1)}

        return {
            "t": round(self.t, 3),
            "status": self.status,
            "lost_for_s": round(self.lost_for_s, 3),
            "pose": None if self.pose is None else {
                "x": round(self.pose.x, 1), "y": round(self.pose.y, 1), "theta": round(self.pose.theta, 4)},
            "arm_left": pt(self.arm_left),
            "arm_right": pt(self.arm_right),
            "warnings": list(self.warnings),
        }


def _project_corners(corners_px: np.ndarray, camera: CameraModel, h_mm: float) -> np.ndarray:
    return np.array([camera.pixel_to_field(c, h_mm) for c in corners_px])


def _marker_point(detections: dict[int, np.ndarray], camera: CameraModel, spec: MarkerSpec) -> Point2D | None:
    if spec.id not in detections:
        return None
    p = _project_corners(detections[spec.id], camera, spec.height_mm).mean(axis=0)
    return Point2D(float(p[0]), float(p[1]))


def _wrap(a: float) -> float:
    """角度を (-pi, pi] に正規化する．"""
    a = math.atan2(math.sin(a), math.cos(a))
    return math.pi if a == -math.pi else a


def _side(p1, p2, q) -> float:
    return (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])


def _crosses_center(points, cfg: Config) -> bool:
    p1, p2 = cfg.center_line
    own = _side(p1, p2, cfg.own_side_point)
    return any(_side(p1, p2, (q.x, q.y)) * own < 0 for q in points if q is not None)


def estimate(detections: dict[int, np.ndarray], camera: CameraModel, cfg: Config,
             prev: PoseResult | None, t: float) -> PoseResult:
    arm_left = _marker_point(detections, camera, cfg.left)
    arm_right = _marker_point(detections, camera, cfg.right)

    if cfg.center.id in detections:
        c = _project_corners(detections[cfg.center.id], camera, cfg.center.height_mm)
        center = c.mean(axis=0)
        # マーカー上方向（ArUco の下辺→上辺）= ロボット前方．4 辺から求めて平均し，ノイズを減らす
        rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        up = (c[0] - c[3]) + (c[1] - c[2]) + rot90 @ (c[1] - c[0]) + rot90 @ (c[2] - c[3])
        pose = Pose2D(float(center[0]), float(center[1]), _wrap(math.atan2(up[1], up[0])))
        status = STATUS_OK
    elif arm_left is not None and arm_right is not None:
        dx, dy = arm_right.x - arm_left.x, arm_right.y - arm_left.y
        pose = Pose2D((arm_left.x + arm_right.x) / 2, (arm_left.y + arm_right.y) / 2,
                      _wrap(math.atan2(dx, -dy)))  # 左→右ベクトルを +90° 回したのが前方
        status = STATUS_ESTIMATED
    else:
        pose = prev.pose if prev is not None else None
        last_seen = prev.last_seen_t if prev is not None else None
        lost_for = 0.0 if last_seen is None else t - last_seen
        warnings = (WARN_CROSS_CENTER,) if _crosses_center([pose, arm_left, arm_right], cfg) else ()
        return PoseResult(t, STATUS_LOST, lost_for, pose, arm_left, arm_right, warnings, last_seen)

    warnings = (WARN_CROSS_CENTER,) if _crosses_center([pose, arm_left, arm_right], cfg) else ()
    return PoseResult(t, status, 0.0, pose, arm_left, arm_right, warnings, t)
