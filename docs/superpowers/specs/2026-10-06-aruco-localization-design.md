# ArUco マーカーによる自己位置推定（プロトタイプ）設計書

- 日付: 2026-10-06
- 対象: 創造工学 2026「ロボット綱引き」用ロボット（防除機型：本体＋左右に展開するアーム）

## 1. 目的

会場天井の固定カメラ映像から，ロボット上の 3 つの ArUco マーカー（本体中心・左アーム端・右アーム端）を検出し，
フィールド座標系 (mm) でのロボット姿勢 (x, y, θ) と左右アーム先端位置を求める．

今回のスコープは **画面表示とログ記録（JSON Lines）まで**．将来ロボット制御（ノートPC → Raspberry Pi への送信）に
拡張しやすい構成にしておく．

### 背景・制約（ルール資料より）

- フィールド: 片面 約 1800×900mm × 2 面．床面は艶消し黒，外周壁は艶消し白（高さ約 300mm），中央壁 高さ 183mm．
- ロボットはスタート時 250×250×300mm 以内，スタート後の変形可（アーム展開）．相手コートへの侵入は反則．
- 把持時は「ロボットやアームの位置を何らかの方法で制御」する必要がある（C-(1)）→ アーム先端位置の実測が根拠になる．
- 天井カメラは会場備え付けで競技中の使用可（教員確認済み）．USB でチームのノート PC に接続する．

## 2. 動作環境

- ノート PC 上の Python 3
- 依存: `opencv-contrib-python`（OpenCV ≥ 4.7, `cv2.aruco.ArucoDetector` API），`numpy`，`pyyaml`，テストに `pytest`

## 3. 座標系

- **フィールド座標系**: 右手系．床面上向きを +Z，単位 mm．+X は `field.width_mm`（長辺），+Y は `field.height_mm`（短辺）の方向．
  - `order_corners` は 4 隅を次の順に並べる: [原点, +X 側, 対角, +Y 側]．
    手順: 4 隅を画面上で反時計回りに並べる（＝上から見て反時計回り，右手系で Z 上向き）→
    最初の辺（原点→+X 側）が長辺になる開始位置 2 通りのうち，原点が画像左下に近い方を選ぶ．
    カメラ映像が鏡像でない前提．
  - 確認画面に原点と軸を描画し，目視で確認できるようにする．
- **θ**: +X 軸から反時計回り（フィールドを上から見て）を正とするラジアン，範囲 (-π, π]．

## 4. 構成

| ファイル | 責務 | 主な公開インターフェース |
|---|---|---|
| `config.yaml` | 設定値 | カメラ番号・解像度，フィールド実寸 `field.width_mm/height_mm`，ArUco 辞書（既定 `DICT_4X4_50`），マーカー ID（中心=0, 左=1, 右=2），各マーカーの高さ `height_mm` と一辺 `size_mm`，中央壁の位置（フィールド座標での直線） |
| `localization/config.py` | YAML 読み込みと型付きデータクラス化 | `load_config(path) -> Config` |
| `localization/field.py` | フィールド四角形の検出と隅の並べ替え，手動クリック指定，JSON 保存/読込 | `detect_field_corners(img) -> np.ndarray(4,2) \| None`，`order_corners(pts)`，`pick_corners_manually(img)`，`save_field(path, corners)`，`load_field(path)` |
| `localization/camera_model.py` | 内部パラメータの読込 or 推定，外部パラメータ推定，画素→高さ h 平面への投影 | `class CameraModel`：`from_field(corners_px, field_size, image_size, intrinsics=None)`，`pixel_to_field(uv, h_mm) -> (x, y)`，`field_to_pixel(xyz)` |
| `localization/markers.py` | ArUco 検出 | `detect_markers(img, detector) -> dict[int, np.ndarray(4,2)]`（ID → 4 隅の画素座標） |
| `localization/pose.py` | ロボット姿勢・アーム先端位置の算出（純粋関数） | `estimate(detections, camera_model, marker_cfg, prev, t) -> PoseResult` |
| `localization/sinks.py` | 出力先の抽象化 | `class Sink`（`write(result)`, `close()`），`JsonlSink(path)` |
| `localization/overlay.py` | カメラ映像への描画と俯瞰図の生成 | `draw_camera_view(img, ...)`，`draw_top_view(result, cfg)` |
| `main.py` | ループ: 取得 → 検出 → 推定 → 表示・記録 | CLI 引数: `--config`, `--recalibrate-field`, `--intrinsics PATH`, `--log PATH` |

## 5. 処理の詳細

### 5.1 フィールド検出（起動時 1 回）

1. 保存済みのフィールド JSON があり，`--recalibrate-field` 指定がなければそれを読み込む．
2. なければ自動検出: グレースケール化 → 大津の二値化（黒領域を前景）→ モルフォロジー処理 → 輪郭抽出 →
   面積最大の輪郭を `approxPolyDP` で 4 点近似．4 点にならない，または面積が画像の 10% 未満なら失敗とする．
   カメラに 2 面とも写る場合は面積最大の 1 面が選ばれるため，自陣でなければ手動指定に切り替える．
3. 自動検出結果を表示し，ユーザーがキー操作で承認（Enter）・手動指定へ切替（m）を選ぶ．
4. 手動指定では，画像上で 4 隅をクリックする（順不同，`order_corners` で並べ替え）．
5. 確定した 4 隅（画素座標）を JSON に保存する．

### 5.2 カメラモデル

- 内部パラメータ:
  - `--intrinsics` でファイル（`camera_matrix`, `dist_coeffs` を含む JSON）が与えられればそれを使う．
  - なければ推定: 主点＝画像中心，正方画素，歪みなしと仮定し，床面ホモグラフィ H（フィールド mm → 画素）から
    焦点距離 f を求める（Zhang の制約 h1ᵀ ω h2 = 0 を用いる）．H が退化して f が求まらない（f² ≤ 0 等）場合は，
    水平画角 60° を仮定した f にフォールバックし，警告を出す．
- 外部パラメータ: フィールド 4 隅（Z=0）と画素座標から `cv2.solvePnP`（`SOLVEPNP_IPPE`）で R, t を求める．
- `pixel_to_field(uv, h)`: 歪み補正した正規化座標からカメラ光線を作り，フィールド座標系の平面 Z=h との交点 (x, y) を返す．

### 5.3 姿勢推定

- 中心マーカー（ID 0）が検出された場合:
  - 4 隅を Z=h₀ 平面へ投影し，その平均を位置 (x, y) とする．
  - 向き θ: マーカー上辺（ArUco の隅 0 → 隅 1）の方向ベクトルに垂直な「マーカー上方向」（隅 3 → 隅 0）をロボット前方とみなし，
    その atan2 とする．マーカーはロボット前方を上にして貼る前提．
  - status = `ok`
- 中心マーカーが無く，左右アーム端（ID 1, 2）が両方見える場合:
  - 位置 = 左右先端の中点，θ = 左→右ベクトルを +90° 回転した方向（前方）．status = `estimated`
- それ以外: 直前の姿勢を保持し status = `lost`，`lost_for_s`（最後に姿勢が得られてからの経過秒）を付ける．
  起動直後で直前姿勢が無ければ `pose` は `null`．
- アーム先端: ID 1, 2 それぞれの 4 隅を各自の高さ平面へ投影した平均．未検出なら `null`．
- 越境警告: 中央壁の直線を越える側にアーム先端・ロボット位置があれば `warnings` に `"cross_center_line"` を入れる．

### 5.4 出力（JSON Lines）

1 フレーム 1 行の JSON．既定の出力先は `logs/pose-YYYYmmdd-HHMMSS.jsonl`．

```json
{"t": 12.345, "status": "ok", "lost_for_s": 0.0,
 "pose": {"x": 512.3, "y": 401.8, "theta": 1.571},
 "arm_left": {"x": 380.1, "y": 402.0}, "arm_right": {"x": 644.6, "y": 401.5},
 "warnings": []}
```

- `t` はループ開始からの秒（単調時計）．数値は mm とラジアン．
- 出力は `Sink` 抽象を通す．今回は `JsonlSink` のみ実装．将来 UDP 送信 Sink を同じインターフェースで追加する．

### 5.5 表示

- カメラ映像: フィールド四角形，原点と軸，検出マーカー，ロボット位置・向き矢印，アーム先端を重ねる．
- 俯瞰図: フィールド座標系を縮尺描画し，ロボットとアーム先端・中央線・警告を表示．
- `q` で終了．

## 6. エラー処理

- カメラが開けない: エラーメッセージを出して終了コード 1．
- フレーム取得失敗: 警告を出してスキップし，連続 30 回失敗したら終了．
- フィールド自動検出失敗: 手動クリック指定に移行．
- 内部パラメータ推定失敗: 画角 60° 仮定にフォールバックし警告．

## 7. テスト方針（pytest）

- 合成画像: 既知のカメラ（K, R, t）でフィールド（黒矩形＋白い周囲）と ArUco マーカー（既知の位置・高さ・向き）を
  射影描画した画像を生成するテスト用ヘルパーを用意する．
- 単体テスト:
  - `order_corners`: 任意順の 4 点が規定順に並ぶ．
  - `detect_field_corners`: 合成画像で真の隅から数 px 以内．
  - `CameraModel`: 既知 K で `pixel_to_field(field_to_pixel(p), h)` が p に戻る．推定 f が真値に近い（±5%）．
  - `pose.estimate`: 中心あり／中心なし左右あり／全欠落 の 3 ケースで status と値が期待どおり．
  - `JsonlSink`: 書いた行が JSON として読み戻せ，スキーマどおり．
- 結合テスト: 合成画像から位置誤差 5mm 以内，向き誤差 1° 以内．

## 8. スコープ外（今回はやらない）

- ノート PC → Raspberry Pi への通信（Sink 拡張で後から追加）
- 経路計画・ロボット制御
- カルマンフィルタ等の平滑化
- 複数ロボット（相手ロボット）の追跡
