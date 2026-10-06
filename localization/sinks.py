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
