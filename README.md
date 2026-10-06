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
