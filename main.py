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
import numpy as np

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


CAMERA_MOVED_PX = 10.0


def _field_to_image_homography(cfg: Config, corners):
    """床面 (mm) → 画素 のホモグラフィ（確認画面の描画用）．"""
    W, H = cfg.field_width_mm, cfg.field_height_mm
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, np.asarray(corners, dtype=np.float32))


def confirm_corners(frame, corners, cfg: Config) -> str:
    """自動検出結果（原点・中央線・自陣側）を表示し，'ok' / 'manual' / 'abort' を返す．"""
    view = frame.copy()
    cv2.polylines(view, [corners.round().astype("int32")], True, (255, 255, 0), 2)
    cv2.circle(view, tuple(int(v) for v in corners[0]), 10, (0, 0, 255), -1)  # 原点
    Hm = _field_to_image_homography(cfg, corners)
    line = cv2.perspectiveTransform(np.array([cfg.center_line], dtype=np.float32), Hm)[0]
    own = cv2.perspectiveTransform(np.array([[cfg.own_side_point]], dtype=np.float32), Hm)[0][0]
    cv2.line(view, tuple(int(v) for v in line[0]), tuple(int(v) for v in line[1]), (0, 0, 255), 3)
    cv2.putText(view, "OWN", tuple(int(v) for v in own), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.putText(view, "Enter: OK / m: manual / Esc: quit  (red=center line, OWN=own side)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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


def calibrate_field(frame, cfg: Config, field_path: str, recalibrate: bool, gui: bool):
    """保存済みの 4 隅を読むか，検出（必要なら手動指定）して保存する．失敗時は None．"""
    image_size = (frame.shape[1], frame.shape[0])
    aspect = cfg.field_width_mm / cfg.field_height_mm
    detected = detect_field_corners(frame, aspect=aspect)
    saved = None if recalibrate else load_field(field_path, image_size)
    if saved is not None:
        print(f"保存済みのフィールドを使用: {field_path}")
        if detected is not None and np.abs(detected - saved).max() > CAMERA_MOVED_PX:
            print("警告: 保存済みの 4 隅と現在の映像がずれています．カメラが動いた可能性があります．"
                  "空のフィールドで --recalibrate-field を実行してください", file=sys.stderr)
        return saved
    if not recalibrate and Path(field_path).exists():
        print(f"保存済みのフィールド（{field_path}）は解像度が違うため検出し直します")
    corners = detected
    if gui:
        choice = confirm_corners(frame, corners, cfg) if corners is not None else "manual"
        if choice == "abort":
            return None
        if choice == "manual":
            corners = pick_corners_manually(frame)
    if corners is None:
        return None
    save_field(field_path, corners, image_size)
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
    corners = calibrate_field(frame, cfg, args.field, args.recalibrate_field, gui)
    if corners is None:
        print("エラー: フィールドの 4 隅を決められませんでした（空のフィールドで --recalibrate-field を試してください）",
              file=sys.stderr)
        return 1

    image_size = (frame.shape[1], frame.shape[0])
    intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics else None
    try:
        camera = CameraModel.from_field(corners, cfg.field_width_mm, cfg.field_height_mm, image_size,
                                        camera_height_mm=cfg.camera_height_mm, intrinsics=intrinsics)
    except (ValueError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
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
