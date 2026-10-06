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


ROBOT = [SceneMarker(0, 700, 350, math.radians(30), 250, 80),
         SceneMarker(1, 570, 575, math.radians(30), 150, 60),
         SceneMarker(2, 830, 125, math.radians(30), 150, 60)]


def run_main(tmp_path, video, field, *extra):
    log = tmp_path / f"pose-{len(list(tmp_path.glob('pose-*')))}.jsonl"
    code = main.run(main.parse_args(["--video", video, "--no-gui", "--log", str(log),
                                     "--field", str(field), *extra]))
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return code, rows


def test_saved_field_from_other_resolution_is_redetected(tmp_path):
    cam = make_camera(tilt_deg=4)
    field = tmp_path / "field.json"
    code, _ = run_main(tmp_path, write_frames(tmp_path / "a", [render(cam, [])]), field)
    assert code == 0
    small = [cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA) for f in (render(cam, []), render(cam, ROBOT))]
    code, rows = run_main(tmp_path, write_frames(tmp_path / "b", small), field)
    assert code == 0
    assert json.loads(field.read_text())["image_size"] == [640, 360]
    assert math.hypot(rows[1]["pose"]["x"] - 700, rows[1]["pose"]["y"] - 350) < 10


def test_warns_when_camera_moved_since_saved_field(tmp_path, capsys):
    field = tmp_path / "field.json"
    run_main(tmp_path, write_frames(tmp_path / "a", [render(make_camera(tilt_deg=4), [])]), field)
    capsys.readouterr()
    run_main(tmp_path, write_frames(tmp_path / "b", [render(make_camera(tilt_deg=8), [])]), field)
    assert "--recalibrate-field" in capsys.readouterr().err


def test_run_returns_1_on_wrong_camera_height_units(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(open("config.yaml", encoding="utf-8").read().replace("height_mm: 2500.0", "height_mm: 2.5"),
                   encoding="utf-8")
    code, _ = run_main(tmp_path, write_frames(tmp_path / "a", [render(make_camera(), [])]),
                       tmp_path / "field.json", "--config", str(cfg))
    assert code == 1
