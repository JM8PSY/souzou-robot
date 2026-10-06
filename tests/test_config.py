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


def test_load_config_rejects_own_side_point_on_center_line(tmp_path):
    import pytest

    text = (ROOT / "config.yaml").read_text(encoding="utf-8").replace(
        "own_side_point: [880.0, 0.0]", "own_side_point: [880.0, 900.0]")
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="own_side_point"):
        load_config(p)
