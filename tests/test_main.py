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
