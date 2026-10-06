# ArUco 自己位置推定 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 天井カメラ映像から ArUco マーカー 3 つ（本体中心・左右アーム端）を検出し，フィールド座標 (mm) でのロボット姿勢 (x, y, θ) とアーム先端位置を求めて，画面表示と JSON Lines 記録を行う．

**Architecture:** 起動時に空フィールドの黒い床面の 4 隅を検出してカメラの外部パラメータを求め（内部パラメータは実測カメラ高さから焦点距離を推定，またはファイルから読込），毎フレーム各マーカーの隅画素を「そのマーカーの高さの水平面」へ光線投影して視差なしの位置を得る．各モジュールは 1 ファイル 1 責務の純粋関数中心で，`main.py` がループで繋ぐ．出力は `Sink` 抽象を通し，今回は JSON Lines のみ．

**Tech Stack:** Python 3.10+，opencv-contrib-python（≥ 4.7，`cv2.aruco.ArucoDetector` API．OpenCV 5.0 で検証済み），numpy，pyyaml，pytest

**Spec:** `docs/superpowers/specs/2026-10-06-aruco-localization-design.md`

## Global Constraints

- Python 3.10 以上（`X | None` 型表記を使用）．依存は `opencv-contrib-python>=4.7`, `numpy`, `pyyaml`, `pytest` のみ．
- 単位は mm とラジアン．フィールド座標は右手系・Z 上向き，θ は +X から反時計回りで (-π, π]．
- フィールド 4 隅の順序は常に [原点, +X 側, 対角, +Y 側]（`order_corners` が保証），対応するフィールド座標は (0,0),(W,0),(W,H),(0,H)．
- マーカー ID 既定: 中心=0，左=1，右=2．ArUco 辞書既定 `DICT_4X4_50`．ArUco の隅順は [左上, 右上, 右下, 左下]，マーカー上方向＝ロボット前方．
- ログは JSON Lines（1 フレーム 1 行），既定パス `logs/pose-YYYYmmdd-HHMMSS.jsonl`．キーは `t, status, lost_for_s, pose{x,y,theta}, arm_left{x,y}, arm_right{x,y}, warnings`．
- status は `ok` / `estimated` / `lost` の 3 種．警告は `cross_center_line` のみ．
- コード中のコメント・ユーザー向けメッセージは日本語．
- テストはすべてリポジトリ直下から `python -m pytest` で実行する（`pytest.ini` で `pythonpath = .`）．

## Review Focus

1. **ロボットが床の縁にかかった状態でのフィールド検出** — 隅が数十 mm ずれる．仕様どおり空フィールドで検出し保存・再利用する運用にし，`main.py` のエラーメッセージで空フィールドでの再校正を促す（Task 8 の `test_run_on_recorded_frames_writes_jsonl` は 1 枚目を空フィールドにして検証）．
2. **画像端に接する暗い領域（会場の床・カメラの黒枠）** — 最大の暗領域が会場床になりフィールドを取り違える．画像端に接する輪郭を除外し，暗い枠に囲まれた床も拾えるよう `RETR_CCOMP` を使う（Task 3 `test_detect_field_ignores_dark_region_touching_image_border`）．
3. **カメラがほぼ真下を向いている** — 床面ホモグラフィだけでは焦点距離が定まらず，IPPE は鏡像解（カメラが床下）を返しうる．ホモグラフィ分解で床の上側の解を選び，カメラ高さで f を決める（Task 2 `test_from_field_with_camera_height_recovers_camera[0.0]`）．
4. **中心マーカーだけ隠れる／片方のアームだけ見える** — 両アームで代替推定，片方だけなら `lost` だがアーム位置は出す（Task 5 `test_estimate_without_center_uses_arms`, `test_estimate_only_one_arm_visible_is_lost_but_reports_arm`）．
5. **起動直後に何も見えない** — `pose: null` で落ちずに記録・描画を続ける（Task 5 `test_estimate_lost_at_startup_has_no_pose`，Task 7 の `LOST_EMPTY` 描画）．

---

## ファイル構成

```
requirements.txt          依存パッケージ
pytest.ini                pytest 設定（pythonpath）
config.yaml               設定値
main.py                   CLI とメインループ
localization/
  __init__.py
  config.py               YAML → 型付き Config
  camera_model.py         CameraModel（外部パラメータ・投影），load_intrinsics
  field.py                床面 4 隅の検出・並べ替え・手動指定・保存
  markers.py              ArUco 検出
  pose.py                 姿勢推定（純粋関数）と PoseResult
  sinks.py                出力先（JsonlSink）
  overlay.py              カメラ映像への描画・俯瞰図
tests/
  __init__.py
  synth.py                合成シーン描画（テスト用ヘルパー）
  test_*.py
```

---
### Task 1: プロジェクト雛形と設定読み込み

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `config.yaml`, `localization/__init__.py`（空）, `tests/__init__.py`（空）
- Create: `localization/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: なし
- Produces: `MarkerSpec(id: int, height_mm: float, size_mm: float)`，`Config`（フィールド: `camera_index, frame_width, frame_height, camera_height_mm, field_width_mm, field_height_mm, center_line: ((x1,y1),(x2,y2)), own_side_point: (x,y), aruco_dictionary: str, center/left/right: MarkerSpec`），`load_config(path) -> Config`

- [ ] **Step 1: 仮想環境と依存を用意する**

`requirements.txt`:

```text
opencv-contrib-python>=4.7
numpy
pyyaml
pytest
```

`pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

`.gitignore`:

```text
__pycache__/
.venv/
.pytest_cache/
logs/
calib/
document/
```

（`document/` には学生名簿を含むルール資料があるためコミットしない．`calib/` は会場ごとの校正結果．）

```bash
python3 -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p localization tests && touch localization/__init__.py tests/__init__.py
```

- [ ] **Step 2: `config.yaml` を作る**

`config.yaml`:

```yaml
camera:
  index: 0
  width: 1280
  height: 720
  height_mm: 2500.0   # 床面からカメラレンズまでの高さ（巻尺で実測）

field:
  width_mm: 1760.0    # +X 方向（長辺）
  height_mm: 900.0    # +Y 方向（短辺）
  # 中央壁の線（フィールド座標）と，自陣側にある任意の 1 点
  center_line:
    p1: [0.0, 900.0]
    p2: [1760.0, 900.0]
  own_side_point: [880.0, 0.0]

aruco:
  dictionary: DICT_4X4_50

markers:
  center: {id: 0, height_mm: 250.0, size_mm: 80.0}
  left:   {id: 1, height_mm: 150.0, size_mm: 60.0}
  right:  {id: 2, height_mm: 150.0, size_mm: 60.0}
```

- [ ] **Step 3: 失敗するテストを書く**

`tests/test_config.py`:

```python
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
```

- [ ] **Step 4: テストが失敗することを確認する**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.config'）

- [ ] **Step 5: 実装を書く — `localization/config.py`**

```python
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
    return Config(
        camera_index=int(cam["index"]),
        frame_width=int(cam["width"]),
        frame_height=int(cam["height"]),
        camera_height_mm=float(cam["height_mm"]),
        field_width_mm=float(field["width_mm"]),
        field_height_mm=float(field["height_mm"]),
        center_line=(_xy(field["center_line"]["p1"]), _xy(field["center_line"]["p2"])),
        own_side_point=_xy(field["own_side_point"]),
        aruco_dictionary=str(raw["aruco"]["dictionary"]),
        center=_marker(markers["center"]),
        left=_marker(markers["left"]),
        right=_marker(markers["right"]),
    )
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS（全件）

- [ ] **Step 7: コミットする**

```bash
git add requirements.txt pytest.ini .gitignore config.yaml localization/__init__.py tests/__init__.py localization/config.py tests/test_config.py
git commit -m "feat: プロジェクト雛形と設定読み込みを追加"
```


### Task 2: カメラモデルと合成シーン（テスト用）

**Files:**
- Create: `localization/camera_model.py`
- Create: `tests/synth.py`（テスト用ヘルパー．以降のタスクでも使う）
- Test: `tests/test_camera_model.py`

**Interfaces:**
- Consumes: なし
- Produces: `field_object_points(field_w, field_h) -> ndarray(4,3)`，`intrinsic_matrix(f, image_size) -> ndarray(3,3)`，`class CameraModel(K, dist, rvec, tvec)`（属性 `K, dist, rvec, tvec, R, center`），`CameraModel.from_corners(corners_px, field_w, field_h, K, dist, refine=True)`，`CameraModel.from_field(corners_px, field_w, field_h, image_size, camera_height_mm=None, intrinsics=None)`，`pixel_to_field(uv, h_mm) -> ndarray(2,)`，`field_to_pixel(xyz) -> ndarray(N,2)`，`load_intrinsics(path) -> (K, dist)`
- Produces (tests/synth.py): `IMAGE_SIZE=(1280,720)`, `FIELD_W=1760.0`, `FIELD_H=900.0`, `CAMERA_HEIGHT=2500.0`, `F_TRUE=900.0`, `make_camera(tilt_deg=5.0, cam_xy=(880,450), height=CAMERA_HEIGHT) -> CameraModel`, `SceneMarker(id, x, y, theta, height_mm, size_mm)`, `marker_world_corners(m) -> ndarray(4,3)`, `render(camera, markers, dictionary='DICT_4X4_50', dark_border_px=0) -> BGR画像`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_camera_model.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_camera_model.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.camera_model'）

- [ ] **Step 3: 実装を書く — `tests/synth.py`**

```python
"""テスト用の合成シーン（天井カメラから見たフィールドとマーカー）を描画する．"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from localization.camera_model import CameraModel

IMAGE_SIZE = (1280, 720)
FIELD_W, FIELD_H = 1760.0, 900.0
CAMERA_HEIGHT = 2500.0
F_TRUE = 900.0


def make_camera(tilt_deg: float = 5.0, cam_xy=(880.0, 450.0), height=CAMERA_HEIGHT) -> CameraModel:
    """真下向きから tilt_deg だけカメラ x 軸回りに傾けた天井カメラ．"""
    # world -> camera：カメラ x = 世界 +X，カメラ y = 世界 -Y，カメラ z = 世界 -Z（真下）
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    a = math.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    R = Rx @ R0
    C = np.array([cam_xy[0], cam_xy[1], height])
    t = -R @ C
    rvec, _ = cv2.Rodrigues(R)
    w, h = IMAGE_SIZE
    K = np.array([[F_TRUE, 0, w / 2], [0, F_TRUE, h / 2], [0, 0, 1]])
    return CameraModel(K, np.zeros(5), rvec, t)


@dataclass
class SceneMarker:
    id: int
    x: float
    y: float
    theta: float  # マーカー上方向（ロボット前方）の向き [rad]
    height_mm: float
    size_mm: float


def marker_world_corners(m: SceneMarker) -> np.ndarray:
    """マーカーの黒い部分の 4 隅（ArUco の順: 左上, 右上, 右下, 左下）を世界座標で返す．"""
    up = np.array([math.cos(m.theta), math.sin(m.theta)])
    right = np.array([math.sin(m.theta), -math.cos(m.theta)])
    c = np.array([m.x, m.y])
    s = m.size_mm / 2
    pts = [c - right * s + up * s, c + right * s + up * s, c + right * s - up * s, c - right * s - up * s]
    return np.array([[p[0], p[1], m.height_mm] for p in pts])


def render(camera: CameraModel, markers: list[SceneMarker], dictionary: str = "DICT_4X4_50",
           dark_border_px: int = 0) -> np.ndarray:
    w, h = IMAGE_SIZE
    img = np.full((h, w, 3), 255, np.uint8)  # 白い壁
    if dark_border_px:
        # 部屋の床など，画像端に接する暗い領域
        img[:dark_border_px, :] = 30
        img[-dark_border_px:, :] = 30
        img[:, :dark_border_px] = 30
        img[:, -dark_border_px:] = 30
    floor = camera.field_to_pixel(
        [[0, 0, 0], [FIELD_W, 0, 0], [FIELD_W, FIELD_H, 0], [0, FIELD_H, 0]]
    )
    cv2.fillConvexPoly(img, np.round(floor).astype(np.int32), (20, 20, 20))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    for m in markers:
        side = 120
        pad = side // 6  # 白いクワイエットゾーン（1 セル分）
        tag = cv2.aruco.generateImageMarker(d, m.id, side)
        tag = cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        total = side + 2 * pad
        padded = SceneMarker(m.id, m.x, m.y, m.theta, m.height_mm, m.size_mm * total / side)
        dst = camera.field_to_pixel(marker_world_corners(padded)).astype(np.float32)
        src = np.array([[0, 0], [total, 0], [total, total], [0, total]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tag, H, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((total, total), 255, np.uint8), H, (w, h))
        img[mask > 127] = warped[mask > 127]
    return img
```

- [ ] **Step 4: 実装を書く — `localization/camera_model.py`**

```python
"""天井カメラのモデル．フィールド 4 隅から外部パラメータを求め，画素を高さ h の平面へ投影する．"""
import math

import cv2
import numpy as np


def field_object_points(field_w: float, field_h: float) -> np.ndarray:
    """order_corners の順 [原点, +X 側, 対角, +Y 側] に対応するフィールド座標 (Z=0)．"""
    return np.array(
        [[0, 0, 0], [field_w, 0, 0], [field_w, field_h, 0], [0, field_h, 0]], dtype=np.float64
    )


def intrinsic_matrix(f: float, image_size: tuple[int, int]) -> np.ndarray:
    """主点=画像中心，正方画素の内部パラメータ行列．"""
    w, h = image_size
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _pose_from_homography(corners_px, obj: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """床面ホモグラフィを分解して (rvec, tvec) を求める（Zhang の方法）．"""
    img = np.asarray(corners_px, dtype=np.float64).reshape(-1, 1, 2)
    n = cv2.undistortPoints(img, K, dist).reshape(-1, 2)
    H, _ = cv2.findHomography(obj[:, :2], n)
    b1, b2, b3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 2.0 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    if b3[2] * lam < 0:  # フィールドがカメラの前方にある解を選ぶ
        lam = -lam
    r1, r2, t = lam * b1, lam * b2, lam * b3
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


class CameraModel:
    def __init__(self, K: np.ndarray, dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        self.R, _ = cv2.Rodrigues(self.rvec)
        # フィールド座標系でのカメラ中心
        self.center = (-self.R.T @ self.tvec).reshape(3)

    @classmethod
    def from_corners(cls, corners_px, field_w: float, field_h: float, K, dist, refine: bool = True) -> "CameraModel":
        obj = field_object_points(field_w, field_h)
        K = np.asarray(K, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        rvec, tvec = _pose_from_homography(corners_px, obj, K, dist)
        if refine:
            img = np.asarray(corners_px, dtype=np.float64).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError("solvePnP に失敗しました")
        return cls(K, dist, rvec, tvec)

    @classmethod
    def from_field(
        cls,
        corners_px,
        field_w: float,
        field_h: float,
        image_size: tuple[int, int],
        camera_height_mm: float | None = None,
        intrinsics: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "CameraModel":
        """intrinsics (K, dist) があればそれを使う．無ければ，実測したカメラ高さに
        合うように焦点距離 f を二分探索で決める（主点=画像中心・正方画素・歪みなしを仮定）．"""
        if intrinsics is not None:
            K, dist = intrinsics
            return cls.from_corners(corners_px, field_w, field_h, K, dist)
        if camera_height_mm is None:
            raise ValueError("intrinsics か camera_height_mm のどちらかが必要です")
        dist = np.zeros(5)
        lo, hi = math.log(50.0), math.log(50000.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            K = intrinsic_matrix(math.exp(mid), image_size)
            model = cls.from_corners(corners_px, field_w, field_h, K, dist, refine=False)
            if model.center[2] < camera_height_mm:
                lo = mid
            else:
                hi = mid
        K = intrinsic_matrix(math.exp((lo + hi) / 2), image_size)
        return cls.from_corners(corners_px, field_w, field_h, K, dist)

    def pixel_to_field(self, uv, h_mm: float) -> np.ndarray:
        """画素 (u, v) を通るカメラ光線と平面 Z=h_mm の交点 (x, y) [mm] を返す．"""
        pt = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
        n = cv2.undistortPoints(pt, self.K, self.dist).reshape(2)
        d = self.R.T @ np.array([n[0], n[1], 1.0])
        s = (h_mm - self.center[2]) / d[2]
        p = self.center + s * d
        return p[:2]

    def field_to_pixel(self, xyz) -> np.ndarray:
        """フィールド座標 (x, y, z) [mm] の点（複数可）を画素座標 (N, 2) に射影する．"""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
        img, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return img.reshape(-1, 2)


def load_intrinsics(path) -> tuple[np.ndarray, np.ndarray]:
    """{"camera_matrix": 3x3, "dist_coeffs": [...]} 形式の JSON を読む．"""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    return K, dist
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `python -m pytest tests/test_camera_model.py -v`
Expected: PASS（全件）

- [ ] **Step 6: 設計上の注意を確認する**

`from_field` は IPPE ではなくホモグラフィ分解 → `SOLVEPNP_ITERATIVE` で精密化する．f が真値からずれると IPPE は鏡像解（カメラが床下）を返し，二分探索が破綻するため（検証済み）．`tilt=0.0` のテストがこれを押さえている．

- [ ] **Step 7: コミットする**

```bash
git add localization/camera_model.py tests/synth.py tests/test_camera_model.py
git commit -m "feat: カメラモデル（高さ平面への投影）と合成シーンを追加"
```


### Task 3: フィールド 4 隅の検出・並べ替え・保存

**Files:**
- Create: `localization/field.py`
- Test: `tests/test_field.py`

**Interfaces:**
- Consumes: `tests/synth.py` の `make_camera`, `render`, `SceneMarker`, `FIELD_W`, `FIELD_H`
- Produces: `order_corners(pts) -> ndarray(4,2)`，`detect_field_corners(img_bgr, min_area_ratio=0.10) -> ndarray(4,2) | None`（order_corners 済み），`save_field(path, corners)`，`load_field(path) -> ndarray(4,2) | None`，`pick_corners_manually(img_bgr, window=...) -> ndarray(4,2) | None`（GUI．単体テスト対象外）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_field.py`:

```python
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
    save_field(path, corners)
    np.testing.assert_allclose(load_field(path), corners)


def test_load_field_missing_returns_none(tmp_path):
    assert load_field(tmp_path / "nope.json") is None
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_field.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.field'）

- [ ] **Step 3: 実装を書く — `localization/field.py`**

```python
"""フィールド（艶消し黒の床面）の 4 隅を検出・並べ替え・保存する．"""
import json
import math
from pathlib import Path

import cv2
import numpy as np


def order_corners(pts) -> np.ndarray:
    """4 隅を [原点, +X 側, 対角, +Y 側] の順に並べる．

    画面上で反時計回りに並べ，最初の辺（原点→+X 側）が長辺になる 2 通りのうち，
    原点が画像左下に近い方を選ぶ．
    """
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    # 画像は y 下向きなので y を反転して角度を取ると，画面上の反時計回りになる
    ang = np.arctan2(-(p[:, 1] - c[1]), p[:, 0] - c[0])
    ccw = p[np.argsort(ang)]
    best, best_score = None, math.inf
    for k in range(4):
        o = np.roll(ccw, -k, axis=0)
        if np.linalg.norm(o[1] - o[0]) < np.linalg.norm(o[3] - o[0]):
            continue
        score = o[0][0] - o[0][1]  # 小さいほど左下（x 小，y 大）
        if score < best_score:
            best, best_score = o, score
    return best


def detect_field_corners(img_bgr: np.ndarray, min_area_ratio: float = 0.10) -> np.ndarray | None:
    """黒い床面の 4 隅（order_corners 済み）を返す．見つからなければ None．"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    # CCOMP: 穴の内側にある領域（暗い枠に囲まれた床面）も外側輪郭として得られる
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, min_area_ratio * w * h
    for cnt, (_, _, _, parent) in zip(contours, hierarchy[0] if hierarchy is not None else []):
        if parent != -1:
            continue  # 穴の輪郭は対象外
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # 画像端に接する領域（部屋の床など）は除外
        area = cv2.contourArea(cnt)
        if area < best_area:
            continue
        hull = cv2.convexHull(cnt)  # マーカーやロボットによる欠けを埋める
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            best, best_area = approx.reshape(4, 2), area
    return None if best is None else order_corners(best)


def save_field(path: str | Path, corners) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"corners_px": np.asarray(corners, dtype=float).reshape(4, 2).tolist()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_field(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["corners_px"], dtype=np.float64).reshape(4, 2)


def pick_corners_manually(img_bgr: np.ndarray, window: str = "field: click 4 corners") -> np.ndarray | None:
    """画像上で 4 隅をクリックしてもらう．Enter で確定，Backspace で 1 点取り消し，Esc で中止．"""
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            view = img_bgr.copy()
            for i, p in enumerate(pts):
                cv2.circle(view, p, 6, (0, 0, 255), -1)
                cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(view, "click 4 corners / Enter: OK / BS: undo / Esc: cancel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (8, 127) and pts:
                pts.pop()
            if key == 13 and len(pts) == 4:
                return order_corners(pts)
    finally:
        cv2.destroyWindow(window)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_field.py -v`
Expected: PASS（全件）

- [ ] **Step 5: コミットする**

```bash
git add localization/field.py tests/test_field.py
git commit -m "feat: フィールド床面の 4 隅検出を追加"
```


### Task 4: ArUco マーカー検出

**Files:**
- Create: `localization/markers.py`
- Test: `tests/test_markers.py`

**Interfaces:**
- Consumes: `tests/synth.py`
- Produces: `create_detector(dictionary: str) -> cv2.aruco.ArucoDetector`（未知の辞書名は ValueError），`detect_markers(img_bgr, detector) -> dict[int, ndarray(4,2)]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_markers.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_markers.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.markers'）

- [ ] **Step 3: 実装を書く — `localization/markers.py`**

```python
"""ArUco マーカーの検出．"""
import cv2
import numpy as np


def create_detector(dictionary: str) -> cv2.aruco.ArucoDetector:
    """辞書名（例: "DICT_4X4_50"）から検出器を作る．"""
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"未知の ArUco 辞書です: {dictionary}")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, params)


def detect_markers(img_bgr: np.ndarray, detector: cv2.aruco.ArucoDetector) -> dict[int, np.ndarray]:
    """ID → 4 隅の画素座標 (4, 2)（左上, 右上, 右下, 左下）を返す．同じ ID が複数あれば最初の 1 つ．"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result: dict[int, np.ndarray] = {}
    if ids is None:
        return result
    for c, i in zip(corners, ids.reshape(-1)):
        result.setdefault(int(i), c.reshape(4, 2).astype(np.float64))
    return result
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_markers.py -v`
Expected: PASS（全件）

- [ ] **Step 5: コミットする**

```bash
git add localization/markers.py tests/test_markers.py
git commit -m "feat: ArUco マーカー検出を追加"
```


### Task 5: 姿勢推定

**Files:**
- Create: `localization/pose.py`
- Test: `tests/test_pose.py`

**Interfaces:**
- Consumes: `CameraModel.pixel_to_field`, `CameraModel.from_field`（Task 2），`Config`, `MarkerSpec`, `load_config`（Task 1），`detect_field_corners`（Task 3），`create_detector`, `detect_markers`（Task 4）
- Produces: 定数 `STATUS_OK='ok'`, `STATUS_ESTIMATED='estimated'`, `STATUS_LOST='lost'`, `WARN_CROSS_CENTER='cross_center_line'`；`Pose2D(x, y, theta)`，`Point2D(x, y)`，`PoseResult(t, status, lost_for_s, pose, arm_left, arm_right, warnings: tuple[str,...], last_seen_t)` と `PoseResult.to_dict() -> dict`（`last_seen_t` は含めない．x,y は小数 1 桁，theta 4 桁，t と lost_for_s 3 桁に丸める）；`estimate(detections, camera, cfg, prev, t) -> PoseResult`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_pose.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_pose.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.pose'）

- [ ] **Step 3: 実装を書く — `localization/pose.py`**

```python
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
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_pose.py -v`
Expected: PASS（全件）

- [ ] **Step 5: コミットする**

```bash
git add localization/pose.py tests/test_pose.py
git commit -m "feat: マーカーからロボット姿勢とアーム先端位置を推定"
```


### Task 6: JSON Lines 出力

**Files:**
- Create: `localization/sinks.py`
- Test: `tests/test_sinks.py`

**Interfaces:**
- Consumes: `PoseResult`, `Pose2D`, `Point2D`（Task 5）
- Produces: `Sink`（Protocol: `write(result)`, `close()`），`JsonlSink(path)`（親ディレクトリを作成，追記モード，毎行 flush）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_sinks.py`:

```python
import json

from localization.pose import Point2D, Pose2D, PoseResult
from localization.sinks import JsonlSink


def test_jsonl_sink_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "logs" / "pose.jsonl"
    sink = JsonlSink(path)
    sink.write(PoseResult(12.34567, "ok", 0.0, Pose2D(512.34, 401.81, 1.570796),
                          Point2D(380.12, 402.0), None, ("cross_center_line",), 12.34567))
    sink.write(PoseResult(12.4, "lost", 0.05, None, None, None, (), None))
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "t": 12.346, "status": "ok", "lost_for_s": 0.0,
        "pose": {"x": 512.3, "y": 401.8, "theta": 1.5708},
        "arm_left": {"x": 380.1, "y": 402.0}, "arm_right": None,
        "warnings": ["cross_center_line"],
    }
    second = json.loads(lines[1])
    assert second["pose"] is None and second["status"] == "lost"
    assert "last_seen_t" not in second
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_sinks.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.sinks'）

- [ ] **Step 3: 実装を書く — `localization/sinks.py`**

```python
"""推定結果の出力先．今回は JSON Lines ファイルのみ．将来 UDP 送信などを同じ形で追加する．"""
import json
from pathlib import Path
from typing import Protocol

from localization.pose import PoseResult


class Sink(Protocol):
    def write(self, result: PoseResult) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """1 フレーム 1 行の JSON を書く．異常終了してもそこまでの行が残るよう毎行 flush する．"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, result: PoseResult) -> None:
        self._f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_sinks.py -v`
Expected: PASS（全件）

- [ ] **Step 5: コミットする**

```bash
git add localization/sinks.py tests/test_sinks.py
git commit -m "feat: 推定結果の JSON Lines 出力を追加"
```


### Task 7: 表示（カメラ映像への描画・俯瞰図）

**Files:**
- Create: `localization/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `CameraModel.field_to_pixel`（Task 2），`Config`（Task 1），`PoseResult`, `STATUS_OK`（Task 5）
- Produces: `draw_camera_view(img_bgr, camera, cfg, corners_px, detections, result) -> 画像`（入力画像は変更しない），`draw_top_view(result, cfg, px_per_mm=0.4, margin=20) -> 画像`（サイズ = (int(H*px_per_mm)+2*margin, int(W*px_per_mm)+2*margin, 3)）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_overlay.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'localization.overlay'）

- [ ] **Step 3: 実装を書く — `localization/overlay.py`**

```python
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
    return out
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: PASS（全件）

- [ ] **Step 5: コミットする**

```bash
git add localization/overlay.py tests/test_overlay.py
git commit -m "feat: カメラ映像への描画と俯瞰図を追加"
```


### Task 8: メインループと CLI

**Files:**
- Create: `main.py`
- Create: `README.md`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: Task 1〜7 のすべての公開関数
- Produces: `parse_args(argv=None) -> Namespace`（`--config`, `--field`, `--recalibrate-field`, `--intrinsics`, `--log`, `--video`, `--no-gui`），`calibrate_field(frame, field_path, recalibrate, gui) -> ndarray | None`，`run(args) -> int`（0=正常，1=エラー）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_main.py`:

```python
import json
import math

import cv2

import main
from tests.synth import SceneMarker, make_camera, render


def write_frames(directory, frames):
    directory.mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(directory / f"{i:04d}.png"), f)
    return str(directory / "%04d.png")


def test_run_on_recorded_frames_writes_jsonl(tmp_path):
    cam = make_camera(tilt_deg=4)
    robot = [SceneMarker(0, 700, 350, math.radians(30), 250, 80),
             SceneMarker(1, 570, 575, math.radians(30), 150, 60),
             SceneMarker(2, 830, 125, math.radians(30), 150, 60)]
    frames = [render(cam, []), render(cam, robot), render(cam, robot[1:]), render(cam, [])]
    video = write_frames(tmp_path / "rec", frames)
    log = tmp_path / "pose.jsonl"
    field = tmp_path / "field.json"

    code = main.run(main.parse_args(["--video", video, "--no-gui", "--log", str(log), "--field", str(field)]))

    assert code == 0
    assert field.exists()  # 空フィールドの 1 枚目で 4 隅を検出して保存
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["lost", "ok", "estimated", "lost"]
    assert abs(rows[1]["pose"]["x"] - 700) < 5 and abs(rows[1]["pose"]["y"] - 350) < 5
    assert rows[3]["pose"] == rows[2]["pose"]  # 見失ったら直前の姿勢を保持


def test_run_returns_1_when_video_cannot_be_opened(tmp_path):
    code = main.run(main.parse_args(["--video", str(tmp_path / "missing.mp4"), "--no-gui",
                                     "--log", str(tmp_path / "x.jsonl"), "--field", str(tmp_path / "f.json")]))
    assert code == 1


def test_run_returns_1_when_field_not_found_without_gui(tmp_path):
    blank = render(make_camera(), []) * 0 + 255  # 床が写っていない
    video = write_frames(tmp_path / "rec", [blank])
    code = main.run(main.parse_args(["--video", video, "--no-gui",
                                     "--log", str(tmp_path / "x.jsonl"), "--field", str(tmp_path / "f.json")]))
    assert code == 1
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'main'）

- [ ] **Step 3: 実装を書く — `main.py`**

```python
"""天井カメラ映像から ArUco マーカーでロボットの自己位置を推定し，表示と JSON Lines 記録を行う．

使い方:
    python main.py                       # config.yaml のカメラで実行
    python main.py --recalibrate-field   # フィールド 4 隅を検出し直す（空のフィールドで行う）
    python main.py --video "rec/%04d.png" --no-gui   # 録画（連番画像・動画）を処理
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from localization.camera_model import CameraModel, load_intrinsics
from localization.config import Config, load_config
from localization.field import detect_field_corners, load_field, pick_corners_manually, save_field
from localization.markers import create_detector, detect_markers
from localization.overlay import draw_camera_view, draw_top_view
from localization.pose import estimate
from localization.sinks import JsonlSink

MAX_CONSECUTIVE_READ_FAILURES = 30
KEY_ENTER, KEY_ESC = 13, 27


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--field", default="calib/field.json", help="フィールド 4 隅の保存先")
    ap.add_argument("--recalibrate-field", action="store_true")
    ap.add_argument("--intrinsics", help="内部パラメータ JSON（camera_matrix, dist_coeffs）")
    ap.add_argument("--log", help="JSON Lines の出力先（既定: logs/pose-日時.jsonl）")
    ap.add_argument("--video", help="カメラの代わりに読む動画ファイル／連番画像パターン")
    ap.add_argument("--no-gui", action="store_true", help="ウィンドウを出さない（自動検出に失敗したら終了）")
    return ap.parse_args(argv)


def open_capture(cfg: Config, video: str | None) -> cv2.VideoCapture:
    if video:
        return cv2.VideoCapture(video)
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    return cap


def confirm_corners(frame, corners) -> str:
    """自動検出結果を表示し，'ok' / 'manual' / 'abort' を返す．"""
    view = frame.copy()
    cv2.polylines(view, [corners.round().astype("int32")], True, (255, 255, 0), 2)
    cv2.circle(view, tuple(int(v) for v in corners[0]), 10, (0, 0, 255), -1)  # 原点
    cv2.putText(view, "Enter: OK / m: manual / Esc: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("field", view)
    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_ENTER:
                return "ok"
            if key == ord("m"):
                return "manual"
            if key == KEY_ESC:
                return "abort"
    finally:
        cv2.destroyWindow("field")


def calibrate_field(frame, field_path: str, recalibrate: bool, gui: bool):
    """保存済みの 4 隅を読むか，検出（必要なら手動指定）して保存する．失敗時は None．"""
    corners = None if recalibrate else load_field(field_path)
    if corners is not None:
        return corners
    corners = detect_field_corners(frame)
    if gui:
        choice = confirm_corners(frame, corners) if corners is not None else "manual"
        if choice == "abort":
            return None
        if choice == "manual":
            corners = pick_corners_manually(frame)
    if corners is None:
        return None
    save_field(field_path, corners)
    return corners


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cap = open_capture(cfg, args.video)
    if not cap.isOpened():
        print(f"エラー: 映像を開けません（{args.video or f'カメラ {cfg.camera_index}'}）", file=sys.stderr)
        return 1
    gui = not args.no_gui

    ok, frame = cap.read()
    if not ok:
        print("エラー: 最初のフレームを取得できません", file=sys.stderr)
        return 1
    corners = calibrate_field(frame, args.field, args.recalibrate_field, gui)
    if corners is None:
        print("エラー: フィールドの 4 隅を決められませんでした（空のフィールドで --recalibrate-field を試してください）",
              file=sys.stderr)
        return 1

    image_size = (frame.shape[1], frame.shape[0])
    intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics else None
    camera = CameraModel.from_field(corners, cfg.field_width_mm, cfg.field_height_mm, image_size,
                                    camera_height_mm=cfg.camera_height_mm, intrinsics=intrinsics)
    print(f"カメラ: f={camera.K[0, 0]:.1f}px, 位置=({camera.center[0]:.0f}, {camera.center[1]:.0f}, "
          f"{camera.center[2]:.0f}) mm")

    detector = create_detector(cfg.aruco_dictionary)
    log_path = args.log or f"logs/pose-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    sink = JsonlSink(log_path)
    print(f"記録先: {log_path}")

    t0 = time.monotonic()
    prev = None
    failures = 0
    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    if args.video:
                        break  # 録画の終端
                    failures += 1
                    print("警告: フレーム取得に失敗しました", file=sys.stderr)
                    if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        print("エラー: フレーム取得に連続で失敗したため終了します", file=sys.stderr)
                        return 1
                    frame = None
                    continue
                failures = 0
            detections = detect_markers(frame, detector)
            result = estimate(detections, camera, cfg, prev, time.monotonic() - t0)
            sink.write(result)
            prev = result
            if gui:
                cv2.imshow("camera", draw_camera_view(frame, camera, cfg, corners, detections, result))
                cv2.imshow("top view", draw_top_view(result, cfg))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame = None
    finally:
        sink.close()
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS（全件）

- [ ] **Step 5: README と全テスト**

`README.md`:

````markdown
# souzou-robot

創造工学 2026「ロボット綱引き」用．天井カメラ映像から ArUco マーカーでロボットの自己位置を推定する．

## セットアップ

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 準備

1. マーカーを印刷する（辞書 `DICT_4X4_50`，ID 0=本体中心，1=左アーム端，2=右アーム端）．周囲に白い余白を残す．
   ロボット前方がマーカーの上になるよう水平に貼る．
2. `config.yaml` を実測値に合わせる: `camera.height_mm`（床からレンズまで），各マーカーの `height_mm`（床から）と `size_mm`（黒枠の一辺），
   `field.center_line` と `own_side_point`．

## 実行

```bash
python main.py --recalibrate-field   # 初回／カメラが動いたとき．ロボットを置かない空のフィールドで行う
python main.py                       # 2 回目以降（calib/field.json を再利用）
```

- 起動時の確認画面: Enter=採用，m=手動で 4 隅をクリック，Esc=中止．赤丸が原点．
- 実行中: `q` で終了．ログは `logs/pose-日時.jsonl`（1 行 1 フレームの JSON）．
- 録画を処理: `python main.py --video "rec/%04d.png" --no-gui`

## テスト

```bash
python -m pytest
```
````

Run: `python -m pytest -v`
Expected: PASS（全テスト．検証時点で 69 件）

- [ ] **Step 6: コミットする**

```bash
git add main.py README.md tests/test_main.py
git commit -m "feat: メインループと CLI を追加"
```

---

## 実機での確認（実装後，会場で）

- [ ] 空のフィールドで `python main.py --recalibrate-field` → 原点（赤丸）と軸が想定どおりか目視確認．
- [ ] 表示される `カメラ: f=..., 位置=(...)` の高さが実測とおおむね一致するか．
- [ ] フィールド上に巻尺で測った 2〜3 点にロボットを置き，`x, y` と実測の差を記録する（目標: 10mm 以内）．
