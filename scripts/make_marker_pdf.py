"""config.yaml のマーカー 3 枚（左・中央・右）を A4 横 1 枚に等間隔で並べた PDF を作る．

印刷時は「実際のサイズ（100%）」で印刷し，下部の 100 mm 目盛りで寸法を確認すること．
使い方: python scripts/make_marker_pdf.py [--config config.yaml] [--out image/markers.pdf]
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization.config import load_config  # noqa: E402

PT_PER_MM = 72.0 / 25.4
PAGE_W_MM, PAGE_H_MM = 297.0, 210.0  # A4 横


def marker_bits(dictionary: str, marker_id: int) -> list[list[bool]]:
    """黒枠込みのセル配列（True = 黒）を返す．"""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    cells = d.markerSize + 2
    img = cv2.aruco.generateImageMarker(d, marker_id, cells, borderBits=1)
    return [[img[r, c] < 128 for c in range(cells)] for r in range(cells)]


def marker_ops(bits: list[list[bool]], x_mm: float, top_mm: float, size_mm: float) -> list[str]:
    """左上 (x_mm, top_mm)・一辺 size_mm のマーカーを描く PDF 演算子列．"""
    n = len(bits)
    cell = size_mm / n * PT_PER_MM
    x0 = x_mm * PT_PER_MM
    y_top = (PAGE_H_MM - top_mm) * PT_PER_MM
    ops = ["0 g"]
    for r, row in enumerate(bits):
        for c, black in enumerate(row):
            if black:
                ops.append(f"{x0 + c * cell:.4f} {y_top - (r + 1) * cell:.4f} {cell:.4f} {cell:.4f} re")
    ops.append("f")
    return ops


def text_op(x_mm: float, y_mm: float, size_pt: float, s: str) -> str:
    return f"BT /F1 {size_pt} Tf {x_mm * PT_PER_MM:.2f} {(PAGE_H_MM - y_mm) * PT_PER_MM:.2f} Td ({s}) Tj ET"


def build_pdf(content: str) -> bytes:
    w, h = PAGE_W_MM * PT_PER_MM, PAGE_H_MM * PT_PER_MM
    stream = content.encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="image/markers.pdf")
    args = ap.parse_args()
    cfg = load_config(args.config)

    specs = [("left", cfg.left), ("center", cfg.center), ("right", cfg.right)]
    total = sum(s.size_mm for _, s in specs)
    gap = (PAGE_W_MM - total) / (len(specs) + 1)  # 余白とマーカー間隔をすべて等しくする
    if gap < 10.0:
        raise SystemExit(f"マーカーが A4 横に収まりません（間隔 {gap:.1f} mm）")
    mid_y = 95.0  # マーカー中心の高さ（ページ上端から mm）

    ops: list[str] = []
    x = gap
    for name, s in specs:
        top = mid_y - s.size_mm / 2
        ops += marker_ops(marker_bits(cfg.aruco_dictionary, s.id), x, top, s.size_mm)
        ops.append(text_op(x, top + s.size_mm + 8, 10, f"{name}  ID {s.id}  {s.size_mm:g} mm"))
        x += s.size_mm + gap

    # 印刷倍率確認用の 100 mm 目盛り
    y = PAGE_H_MM - 20
    sx = (PAGE_W_MM - 100) / 2
    ops.append("0.5 w 0 G")
    ops.append(f"{sx * PT_PER_MM:.2f} {(PAGE_H_MM - y) * PT_PER_MM:.2f} m {(sx + 100) * PT_PER_MM:.2f} {(PAGE_H_MM - y) * PT_PER_MM:.2f} l S")
    for i in range(11):
        tx = (sx + i * 10) * PT_PER_MM
        tick = 3 if i % 5 else 5
        ops.append(f"{tx:.2f} {(PAGE_H_MM - y) * PT_PER_MM:.2f} m {tx:.2f} {(PAGE_H_MM - y + tick) * PT_PER_MM:.2f} l S")
    ops.append(text_op(sx, y + 6, 8, f"100 mm  /  {cfg.aruco_dictionary}  /  print at 100% (actual size)"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(build_pdf("\n".join(ops)))
    print(f"{args.out} を書き出しました（間隔 {gap:.2f} mm）")


if __name__ == "__main__":
    main()
