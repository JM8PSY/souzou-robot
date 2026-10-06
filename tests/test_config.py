from pathlib import Path

from localization.config import MarkerSpec, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_load_default_config():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.camera_index == 0
    assert (cfg.frame_width, cfg.frame_height) == (1280, 720)
    assert cfg.camera_height_mm == 2500.0
    assert (cfg.field_width_mm, cfg.field_height_mm) == (1760.0, 900.0)
    assert cfg.center_line == ((0.0, 900.0), (1760.0, 900.0))
    assert cfg.own_side_point == (880.0, 0.0)
    assert cfg.aruco_dictionary == "DICT_4X4_50"
    assert cfg.center == MarkerSpec(id=0, height_mm=250.0, size_mm=80.0)
    assert cfg.left.id == 1 and cfg.right.id == 2
