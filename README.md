# MILo Video to 3D Web App

動画をアップロードし、MILo を使って動画内の対象を 3D メッシュとして再構成する FastAPI ベースの Web アプリです。

## 出力形式

- USD ASCII (`.usda`)
- OBJ (`.obj`)

アップロードした動画ファイルは `output` に保存しません。生成された 3D モデルだけを保存します。

FBX は現在の Python 依存関係では直接エクスポートできないため未対応です。必要な場合は、生成した OBJ または USD を Blender などで FBX に変換してください。

## 変換キュー

MILo の変換処理は重いため、同時実行せずキューで1件ずつ処理します。

複数の変換依頼が同時に来た場合、後から来た依頼はキューで待機し、前の処理が終わってから順番に実行されます。これにより、GPU、CPU、COLMAP、MILo の外部プロセスが同時に走って不安定になることを防ぎます。

## MILo の自動セットアップ

`start.bat` 実行時に環境チェックを行います。`MILO_ROOT` が未設定で MILo が見つからない場合、このアプリは自動で以下を実行します。

1. `./.milo/MILo` に `Anttwo/MILo` を `git clone --recursive`
2. MILo の `install.py` を実行
3. COLMAP がない場合、Windows では `winget` でのインストールを試行
4. `winget` で見つからない場合、COLMAP公式GitHub ReleaseのWindows zipを `./.tools/COLMAP` に取得して使用
5. Web サーバーを起動

自動セットアップを止めたい場合:

```bat
set MILO_AUTO_INSTALL=0
```

既存の MILo 環境を使いたい場合:

```bat
set MILO_ROOT=C:\path\to\MILo
set MILO_PYTHON=C:\path\to\milo-env\python.exe
```

`MILO_PYTHON` を省略した場合は、このアプリを実行している Python を使います。

CUDA バージョンを指定したい場合:

```bat
set MILO_CUDA_VERSION=12.1
```

MILo のインストール済み環境をそのまま使い、`install.py` を実行したくない場合:

```bat
set MILO_SKIP_INSTALL=1
```

COLMAP の自動インストールだけを止めたい場合:

```bat
set COLMAP_AUTO_INSTALL=0
```

## 別途必要になる場合があるもの

COLMAP は OS 側の実行ファイルです。Windows ではまず `winget` による自動インストールを試し、見つからない場合はCOLMAP公式GitHub ReleaseのWindows zipを `./.tools/COLMAP` に展開して、このアプリの実行中PATHへ追加します。

## 起動

```bat
start.bat
```

ブラウザで表示された URL を開きます。標準では `http://127.0.0.1:8765/` です。

## 処理の流れ

1. 変換依頼をキューに投入
2. キューの先頭から1件ずつ処理
3. 動画から `dataset/input/*.jpg` を抽出
4. MILo の `convert.py` で COLMAP データセット化
5. MILo の `train.py` で学習
6. MILo の `mesh_extract_sdf.py` で PLY メッシュを生成
7. このアプリ側で PLY を USD または OBJ に変換
