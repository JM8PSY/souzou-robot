import math

import numpy as np
import pytest

from localization.markers import create_detector, detect_markers
from tests.synth import SceneMarker, make_camera, marker_world_corners, render


def test_detect_markers_finds_ids_and_corners():
    cam = make_camera()
    markers = [SceneMarker(0, 880, 450, math.radians(30), 250, 80),
               SceneMarker(1, 500, 300, 0.0, 150, 60),
               SceneMarker(2, 1300, 600, math.pi, 150, 60)]
    found = detect_markers(render(cam, markers), create_detector("DICT_4X4_50"))
    assert set(found) == {0, 1, 2}
    for m in markers:
        expected = cam.field_to_pixel(marker_world_corners(m))
        np.testing.assert_allclose(found[m.id], expected, atol=1.5)


def test_detect_markers_empty_image():
    img = np.full((720, 1280, 3), 255, np.uint8)
    assert detect_markers(img, create_detector("DICT_4X4_50")) == {}


def test_create_detector_rejects_unknown_dictionary():
    with pytest.raises(ValueError):
        create_detector("DICT_NOPE")
