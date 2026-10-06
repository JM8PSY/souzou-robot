"""config.yaml を読み込み，型付きの設定オブジェクトにする．"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MarkerSpec:
    id: int
    height_mm: float
    size_mm: float


@dataclass(frozen=True)
class Config:
    camera_index: int
    frame_width: int
    frame_height: int
    camera_height_mm: float
    field_width_mm: float
    field_height_mm: float
    center_line: tuple[tuple[float, float], tuple[float, float]]
    own_side_point: tuple[float, float]
    aruco_dictionary: str
    center: MarkerSpec
    left: MarkerSpec
    right: MarkerSpec


def _marker(d: dict) -> MarkerSpec:
    return MarkerSpec(id=int(d["id"]), height_mm=float(d["height_mm"]), size_mm=float(d["size_mm"]))


def _xy(v) -> tuple[float, float]:
    return (float(v[0]), float(v[1]))


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cam, field, markers = raw["camera"], raw["field"], raw["markers"]
    center_line = (_xy(field["center_line"]["p1"]), _xy(field["center_line"]["p2"]))
    own = _xy(field["own_side_point"])
    (x1, y1), (x2, y2) = center_line
    if abs((x2 - x1) * (own[1] - y1) - (y2 - y1) * (own[0] - x1)) < 1e-6:
        raise ValueError("field.own_side_point が中央線上にあります．自陣側の点を指定してください")
    return Config(
        camera_index=int(cam["index"]),
        frame_width=int(cam["width"]),
        frame_height=int(cam["height"]),
        camera_height_mm=float(cam["height_mm"]),
        field_width_mm=float(field["width_mm"]),
        field_height_mm=float(field["height_mm"]),
        center_line=center_line,
        own_side_point=own,
        aruco_dictionary=str(raw["aruco"]["dictionary"]),
        center=_marker(markers["center"]),
        left=_marker(markers["left"]),
        right=_marker(markers["right"]),
    )
