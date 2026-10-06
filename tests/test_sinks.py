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
