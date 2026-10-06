import itertools

import numpy as np
import pytest

from localization.field import detect_field_corners, load_field, order_corners, save_field
from tests.synth import FIELD_H, FIELD_W, SceneMarker, make_camera, render

FIELD_PTS = [[0, 0, 0], [FIELD_W, 0, 0], [FIELD_W, FIELD_H, 0], [0, FIELD_H, 0]]

# 画像上の長方形（横長）：左下, 右下, 右上, 左上
BL, BR, TR, TL = (100, 600), (1100, 600), (1100, 100), (100, 100)


@pytest.mark.parametrize("perm", list(itertools.permutations([BL, BR, TR, TL])))
def test_order_corners_landscape_any_input_order(perm):
    np.testing.assert_allclose(order_corners(perm), [BL, BR, TR, TL])


def test_order_corners_portrait_long_side_is_x():
    # 縦長に写っている（カメラが 90° 回っている）場合，+X は長辺＝画面の縦方向になる
    bl, br, tr, tl = (300, 1000), (700, 1000), (700, 100), (300, 100)
    o = order_corners([tl, br, bl, tr])
    edge_x = np.linalg.norm(o[1] - o[0])
    edge_y = np.linalg.norm(o[3] - o[0])
    assert edge_x > edge_y
    # 画面上で反時計回り（y 反転後の符号付き面積が正）
    x, y = o[:, 0], -o[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


@pytest.mark.parametrize("tilt", [0.0, 10.0])
def test_detect_field_corners_synthetic(tilt):
    cam = make_camera(tilt_deg=tilt)
    img = render(cam, [SceneMarker(0, 880, 450, 1.0, 250, 80)])
    found = detect_field_corners(img)
    assert found is not None
    np.testing.assert_allclose(found, cam.field_to_pixel(FIELD_PTS), atol=3.0)


def test_detect_field_ignores_dark_region_touching_image_border():
    cam = make_camera()
    img = render(cam, [], dark_border_px=40)
    found = detect_field_corners(img)
    assert found is not None
    np.testing.assert_allclose(found, cam.field_to_pixel(FIELD_PTS), atol=3.0)


def test_detect_field_returns_none_on_blank_image():
    assert detect_field_corners(np.full((720, 1280, 3), 255, np.uint8)) is None


def test_save_and_load_field(tmp_path):
    corners = np.array([BL, BR, TR, TL], dtype=float)
    path = tmp_path / "sub" / "field.json"
    save_field(path, corners, (1280, 720))
    np.testing.assert_allclose(load_field(path, (1280, 720)), corners)


def test_load_field_rejects_different_image_size(tmp_path):
    """解像度が変わった（カメラの設定失敗・別カメラ）なら保存済みの 4 隅は使わない．"""
    path = tmp_path / "field.json"
    save_field(path, np.array([BL, BR, TR, TL], dtype=float), (1280, 720))
    assert load_field(path, (640, 360)) is None


def test_load_field_missing_returns_none(tmp_path):
    assert load_field(tmp_path / "nope.json", (1280, 720)) is None


def test_detect_field_does_not_merge_two_adjacent_courts():
    """中央壁（幅 20mm）を挟んだ 2 面が 1 つの四角形に融合しないこと．どちらか 1 面を返す．"""
    import cv2

    cam = make_camera(tilt_deg=0.0, cam_xy=(880.0, 910.0), height=3600.0)
    img = render(cam, [])
    court_b = [[0, 920, 0], [FIELD_W, 920, 0], [FIELD_W, 920 + FIELD_H, 0], [0, 920 + FIELD_H, 0]]
    cv2.fillConvexPoly(img, np.round(cam.field_to_pixel(court_b)).astype(np.int32), (20, 20, 20))
    found = detect_field_corners(img, aspect=FIELD_W / FIELD_H)
    assert found is not None
    a = order_corners(cam.field_to_pixel(FIELD_PTS))
    b = order_corners(cam.field_to_pixel(court_b))
    assert np.abs(found - a).max() < 3.0 or np.abs(found - b).max() < 3.0
