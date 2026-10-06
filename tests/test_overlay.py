from pathlib import Path

import numpy as np

from localization.config import load_config
from localization.overlay import draw_camera_view, draw_top_view
from localization.pose import Point2D, Pose2D, PoseResult
from tests.synth import FIELD_H, FIELD_W, make_camera, render

CFG = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
RESULT = PoseResult(1.0, "ok", 0.0, Pose2D(600, 400, 0.3), Point2D(500, 600), None, ("cross_center_line",), 1.0)
LOST_EMPTY = PoseResult(0.0, "lost", 0.0, None, None, None, (), None)


def test_draw_camera_view_returns_annotated_copy():
    cam = make_camera()
    img = render(cam, [])
    corners = cam.field_to_pixel([[0, 0, 0], [FIELD_W, 0, 0], [FIELD_W, FIELD_H, 0], [0, FIELD_H, 0]])
    dets = {0: cam.field_to_pixel([[560, 440, 250], [640, 440, 250], [640, 360, 250], [560, 360, 250]])}
    for r in (RESULT, LOST_EMPTY):
        out = draw_camera_view(img, cam, CFG, corners, dets, r)
        assert out.shape == img.shape
        assert not np.array_equal(out, img)
    assert np.array_equal(img, render(cam, []))  # 元画像は変更しない


def test_draw_top_view_size_matches_field_scale():
    out = draw_top_view(RESULT, CFG, px_per_mm=0.5, margin=10)
    assert out.shape == (int(FIELD_H * 0.5) + 20, int(FIELD_W * 0.5) + 20, 3)
    draw_top_view(LOST_EMPTY, CFG)
