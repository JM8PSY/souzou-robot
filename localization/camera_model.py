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
