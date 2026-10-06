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
