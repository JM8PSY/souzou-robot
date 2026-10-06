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


def _aspect_of(quad: np.ndarray) -> float:
    """向かい合う辺の平均どうしの比（長い方 / 短い方）．"""
    q = order_corners(quad)
    e = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    return max(a, b) / max(min(a, b), 1e-9)


def detect_field_corners(img_bgr: np.ndarray, min_area_ratio: float = 0.10,
                         aspect: float | None = None, aspect_tol: float = 0.30) -> np.ndarray | None:
    """黒い床面の 4 隅（order_corners 済み）を返す．見つからなければ None．

    aspect（= 長辺 / 短辺，例 1760/900）を与えると，縦横比が aspect_tol 以上ずれる四角形
    （隣のコートと融合したもの等）を除外する．"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 大きいと中央壁の細い白線が埋まる
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
        if len(approx) != 4:
            continue
        if aspect is not None and abs(_aspect_of(approx.reshape(4, 2)) / aspect - 1) > aspect_tol:
            continue
        best, best_area = approx.reshape(4, 2), area
    return None if best is None else order_corners(best)


def save_field(path: str | Path, corners, image_size: tuple[int, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"image_size": [int(image_size[0]), int(image_size[1])],
            "corners_px": np.asarray(corners, dtype=float).reshape(4, 2).tolist()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_field(path: str | Path, image_size: tuple[int, int]) -> np.ndarray | None:
    """保存済みの 4 隅を読む．ファイルが無い，または保存時と画像サイズが違えば None．"""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data.get("image_size", ())) != tuple(image_size):
        return None
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
