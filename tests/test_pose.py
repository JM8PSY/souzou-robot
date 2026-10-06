import math
from pathlib import Path

import pytest

from localization.config import load_config
from localization.markers import create_detector, detect_markers
from localization.pose import (STATUS_ESTIMATED, STATUS_LOST, STATUS_OK, WARN_CROSS_CENTER,
                               Pose2D, PoseResult, estimate)
from tests.synth import SceneMarker, make_camera, marker_world_corners, render

CFG = load_config(Path(__file__).resolve().parent.parent / "config.yaml")


def robot_markers(x, y, theta, arm_half_span=300.0):
    """ロボット中心 (x, y)・向き theta に対し，中心と左右アーム端のマーカーを置く．"""
    lx = x + arm_half_span * math.cos(theta + math.pi / 2)
    ly = y + arm_half_span * math.sin(theta + math.pi / 2)
    rx = x + arm_half_span * math.cos(theta - math.pi / 2)
    ry = y + arm_half_span * math.sin(theta - math.pi / 2)
    return {
        "center": SceneMarker(CFG.center.id, x, y, theta, CFG.center.height_mm, CFG.center.size_mm),
        "left": SceneMarker(CFG.left.id, lx, ly, theta, CFG.left.height_mm, CFG.left.size_mm),
        "right": SceneMarker(CFG.right.id, rx, ry, theta, CFG.right.height_mm, CFG.right.size_mm),
    }


def exact_detections(cam, markers):
    """画像を介さず，真の画素座標をそのまま検出結果とする．"""
    return {m.id: cam.field_to_pixel(marker_world_corners(m)) for m in markers}


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@pytest.mark.parametrize("theta", [0.0, math.radians(37), math.pi / 2, math.radians(-150)])
def test_estimate_with_center_marker(theta):
    cam = make_camera(tilt_deg=8)
    ms = robot_markers(600, 400, theta)
    r = estimate(exact_detections(cam, ms.values()), cam, CFG, None, t=1.0)
    assert r.status == STATUS_OK
    assert r.pose.x == pytest.approx(600, abs=0.5)
    assert r.pose.y == pytest.approx(400, abs=0.5)
    assert angle_diff(r.pose.theta, theta) < math.radians(0.1)
    assert (r.arm_left.x, r.arm_left.y) == pytest.approx((ms["left"].x, ms["left"].y), abs=0.5)
    assert (r.arm_right.x, r.arm_right.y) == pytest.approx((ms["right"].x, ms["right"].y), abs=0.5)
    assert r.last_seen_t == 1.0 and r.lost_for_s == 0.0 and r.warnings == ()


def test_estimate_without_center_uses_arms():
    cam = make_camera(tilt_deg=8)
    theta = math.radians(60)
    ms = robot_markers(900, 450, theta)
    r = estimate(exact_detections(cam, [ms["left"], ms["right"]]), cam, CFG, None, t=2.0)
    assert r.status == STATUS_ESTIMATED
    assert (r.pose.x, r.pose.y) == pytest.approx((900, 450), abs=0.5)
    assert angle_diff(r.pose.theta, theta) < math.radians(0.1)


def test_estimate_lost_keeps_previous_pose():
    cam = make_camera()
    prev = PoseResult(1.0, STATUS_OK, 0.0, Pose2D(100, 200, 0.5), None, None, (), 1.0)
    r = estimate({}, cam, CFG, prev, t=3.5)
    assert r.status == STATUS_LOST
    assert r.pose == Pose2D(100, 200, 0.5)
    assert r.lost_for_s == pytest.approx(2.5)
    assert r.last_seen_t == 1.0


def test_estimate_lost_at_startup_has_no_pose():
    r = estimate({}, make_camera(), CFG, None, t=0.2)
    assert r.status == STATUS_LOST and r.pose is None and r.lost_for_s == 0.0


def test_estimate_only_one_arm_visible_is_lost_but_reports_arm():
    cam = make_camera()
    ms = robot_markers(600, 400, 0.0)
    r = estimate(exact_detections(cam, [ms["left"]]), cam, CFG, None, t=1.0)
    assert r.status == STATUS_LOST
    assert r.arm_left is not None and r.arm_right is None


def test_warning_when_arm_crosses_center_line():
    # 既定設定: 中央線 y=900，自陣は y<900．theta=0（+X 向き）で y=700 に置くと，
    # 左アーム端は y=1000 になり中央線を越える
    cam = make_camera()
    ms = robot_markers(880, 700, 0.0)
    r = estimate(exact_detections(cam, ms.values()), cam, CFG, None, t=1.0)
    assert WARN_CROSS_CENTER in r.warnings


def test_no_warning_inside_own_side():
    cam = make_camera()
    ms = robot_markers(880, 400, 0.0)
    r = estimate(exact_detections(cam, ms.values()), cam, CFG, None, t=1.0)
    assert r.warnings == ()


@pytest.mark.parametrize("tilt", [0.0, 6.0])
@pytest.mark.parametrize("x,y,deg", [(700, 350, 25), (150, 800, 90), (1650, 120, -45), (1500, 700, 170)])
def test_end_to_end_synthetic_image_accuracy(tilt, x, y, deg):
    """空フィールドで校正 → ロボットを置いた画像から姿勢推定．位置 5mm・向き 1° 以内．"""
    from localization.camera_model import CameraModel
    from localization.field import detect_field_corners
    from tests.synth import CAMERA_HEIGHT, FIELD_H, FIELD_W, IMAGE_SIZE

    true_cam = make_camera(tilt_deg=tilt)
    corners = detect_field_corners(render(true_cam, []))
    cam = CameraModel.from_field(corners, FIELD_W, FIELD_H, IMAGE_SIZE, camera_height_mm=CAMERA_HEIGHT)
    theta = math.radians(deg)
    ms = robot_markers(x, y, theta, arm_half_span=250)
    img = render(true_cam, list(ms.values()))
    r = estimate(detect_markers(img, create_detector(CFG.aruco_dictionary)), cam, CFG, None, t=0.0)
    assert r.status == STATUS_OK
    assert math.hypot(r.pose.x - x, r.pose.y - y) < 5.0
    assert angle_diff(r.pose.theta, theta) < math.radians(1.0)
