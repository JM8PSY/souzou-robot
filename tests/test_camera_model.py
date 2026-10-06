import numpy as np
import pytest

from localization.camera_model import CameraModel
from tests.synth import CAMERA_HEIGHT, F_TRUE, FIELD_H, FIELD_W, IMAGE_SIZE, make_camera

FIELD_PTS = [[0, 0, 0], [FIELD_W, 0, 0], [FIELD_W, FIELD_H, 0], [0, FIELD_H, 0]]


@pytest.mark.parametrize("h", [0.0, 150.0, 250.0])
def test_pixel_to_field_roundtrip(h):
    cam = make_camera()
    for x, y in [(100.0, 100.0), (880.0, 450.0), (1700.0, 850.0)]:
        uv = cam.field_to_pixel([x, y, h])[0]
        np.testing.assert_allclose(cam.pixel_to_field(uv, h), [x, y], atol=1e-6)


@pytest.mark.parametrize("tilt", [0.0, 5.0, 15.0])
def test_from_field_with_camera_height_recovers_camera(tilt):
    true_cam = make_camera(tilt_deg=tilt)
    corners = true_cam.field_to_pixel(FIELD_PTS)
    cam = CameraModel.from_field(corners, FIELD_W, FIELD_H, IMAGE_SIZE, camera_height_mm=CAMERA_HEIGHT)
    assert cam.K[0, 0] == pytest.approx(F_TRUE, rel=0.01)
    np.testing.assert_allclose(cam.center, true_cam.center, atol=5.0)


def test_from_field_with_intrinsics_uses_them():
    true_cam = make_camera()
    corners = true_cam.field_to_pixel(FIELD_PTS)
    cam = CameraModel.from_field(corners, FIELD_W, FIELD_H, IMAGE_SIZE, intrinsics=(true_cam.K, true_cam.dist))
    np.testing.assert_allclose(cam.K, true_cam.K)
    np.testing.assert_allclose(cam.center, true_cam.center, atol=1e-3)


def test_height_correction_removes_parallax():
    """高さ 250mm の点を床面 (h=0) で投影するとずれるが，h=250 なら正しい位置になる．"""
    cam = make_camera()
    uv = cam.field_to_pixel([1600.0, 800.0, 250.0])[0]
    naive = cam.pixel_to_field(uv, 0.0)
    corrected = cam.pixel_to_field(uv, 250.0)
    assert np.linalg.norm(naive - [1600.0, 800.0]) > 50.0
    np.testing.assert_allclose(corrected, [1600.0, 800.0], atol=1e-6)


def test_from_field_requires_height_or_intrinsics():
    corners = make_camera().field_to_pixel(FIELD_PTS)
    with pytest.raises(ValueError):
        CameraModel.from_field(corners, FIELD_W, FIELD_H, IMAGE_SIZE)


def test_load_intrinsics(tmp_path):
    import json

    from localization.camera_model import load_intrinsics

    p = tmp_path / "intr.json"
    p.write_text(json.dumps({"camera_matrix": [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
                             "dist_coeffs": [0.1, -0.05, 0, 0, 0]}))
    K, dist = load_intrinsics(p)
    assert K.shape == (3, 3) and K[0, 0] == 900
    np.testing.assert_allclose(dist, [0.1, -0.05, 0, 0, 0])
