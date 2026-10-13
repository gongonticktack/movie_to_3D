# MILo Video to 3D（movie_to_3D）

## 概要

動画からフレームを抽出し、COLMAPとMILoで3D形状を再構成して、OBJまたはUSD ASCII（USDA）として保存するローカルWebアプリです。ブラウザで動画と出力形式を指定できます。

## 主な機能

- 動画のアップロードとフレーム抽出。
- COLMAPによるカメラ・シーン推定とMILoによるメッシュ生成。
- OBJ / USDAの出力とダウンロード。
- GPUに合うPyTorch・CUDA環境の選択。
- 変換をキューに入れて1件ずつ順番に処理。

## 推奨要件

| 項目 | 環境 |
| --- | --- |
| OS | Windows 64-bit（自動セットアップの対象） |
| Python | 3.10または3.11 |
| GPU | NVIDIA CUDA対応GPUとドライバー。`nvidia-smi`が利用可能 |
| 開発ツール | Git、CUDA Toolkit、Visual Studio C++ Build Tools |
| 外部ツール | COLMAP |
| 通信・容量 | 初回ダウンロード用の通信と、環境・学習・出力用の十分な空き容量 |

必要VRAM・処理時間は動画と再構成条件に依存します。最低値は本リポジトリで定義していません。GPUのCompute Capabilityに応じてCUDA 11.8 / 12.1 / 12.8系を選ぶ実装です。正確な組合せは`src/runtime_plan.py`を参照してください。

## 必要なライブラリ

- Web：FastAPI、Uvicorn、python-multipart。
- 画像・形状：NumPy、OpenCV、Pillow、trimesh、usd-core。
- 設定・テスト：PyYAML、pytest。
- MILo：CUDA対応PyTorch、torchvision、torchaudioとMILo側のPython / CUDA拡張依存関係。

Web側の依存一覧は`requirements.txt`です。これだけではMILoのGPU再構成環境は完成しません。起動スクリプトと環境確認処理が追加依存を準備します。

## 使い方

### Windowsで起動

1. Python 3.10 / 3.11、Git、NVIDIAドライバーを準備します。
2. ルートの`start.bat`をダブルクリックします。
3. 初回は仮想環境、Python依存関係、GPUに合うPyTorch・CUDA、C++ビルド環境、MILo・COLMAPの準備を待ちます。導入には権限確認が表示される場合があります。
4. 表示されたURLを開きます。既定は `http://127.0.0.1:8765/` です。使用中なら8765から20番先までの空きポートを探します。
5. 動画、シーン設定、出力形式を指定して変換します。
6. 結果をダウンロードします。`output/`にも生成物が保存されます。

終了は起動コンソールで`Ctrl+C`を押します。アップロード元動画は`output/`には保存しません。

### 準備済み環境で手動起動

```bash
python -m pip install -r requirements.txt
python -m src.preflight
python -m uvicorn src.main:app --host 127.0.0.1 --port 8765
```

MILo・CUDA・COLMAPを先に準備した環境で使用してください。

## 設定

| 環境変数 | 内容 |
| --- | --- |
| `PORT` | Windows起動時の開始ポート。既定8765 |
| `MILO_ROOT` | 既存MILoの配置先。未指定は`.milo/MILo` |
| `MILO_PYTHON` | MILoを実行するPython |
| `MILO_AUTO_INSTALL` | MILo自動導入。既定で有効 |
| `MILO_SKIP_INSTALL` | 準備済みMILoの依存導入を省略 |
| `COLMAP_AUTO_INSTALL` | WindowsのCOLMAP自動導入 |
| `CUDA_HOME / CUDA_PATH` | CUDA Toolkitの場所 |
| `MILO_TORCH_CUDA` | 実行環境指定。GPUとの適合検証あり |

## フォルダ構成

| 相対パス | 役割 |
| --- | --- |
| `src/main.py` | Web画面、変換キュー、再構成・出力処理 |
| `src/runtime_plan.py` | GPUとPyTorch / CUDAの組合せ選択 |
| `src/preflight.py` | 実行前の環境確認 |
| `src/fused_ssim.py` | SSIM関連の補助 |
| `tests/` | 変換処理のテスト |
| `output/` | 生成したOBJ・USDA |
| `.milo/` | MILoのローカル配置 |
| `.tools/` | COLMAPなど外部ツール |
| `.venv/` | Python仮想環境 |
| `requirements.txt` | Web側Python依存関係 |
| `start.bat` | Windowsセットアップ・起動 |

## 仕組み

動画から静止画を取り出し、画像間の位置関係をCOLMAPで推定します。MILoが画像とカメラ情報から形状を学習し、メッシュを生成します。最後にOBJまたはUSDAへ変換して返します。複数の依頼は同時処理せず、受付順で実行します。

## 確認方法

```bash
python -m pytest
```

## トラブル対処

- CUDA不一致：`src/runtime_plan.py`の選択結果、ToolkitとPyTorchのCUDA版を確認します。
- CUDA拡張のビルド失敗：C++ Build Tools、CUDA Toolkit、Python版を確認します。
- MILo未検出：`MILO_ROOT`と`train.py`、`mesh_extract_sdf.py`の配置を確認します。
- 再構成が崩れる：被写体の周囲を撮影し、十分な重なり・明るさ・模様がある動画を使用します。

MILoの利用条件は[公式リポジトリ](https://github.com/Anttwo/MILo)を確認してください。
