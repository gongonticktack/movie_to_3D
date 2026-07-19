import argparse
import sys

from src.main import MiloConfigurationError, ensure_milo_ready


def main() -> int:
    """start.bat から呼ばれる起動前チェックを実行する。"""
    # 起動前チェックだけでWebサーバーを落とすかどうかを、必要に応じて切り替えられるようにする。
    parser = argparse.ArgumentParser(description="MILo Webアプリの実行環境を確認し、不足があれば準備します。")
    parser.add_argument(
        "--allow-missing-colmap",
        action="store_true",
        help="COLMAPが見つからない場合も警告扱いにして、Webサーバーの起動だけは許可します。",
    )
    args = parser.parse_args()

    print("MILo環境を確認しています...")
    try:
        # 通常起動ではCOLMAPまで含めて確認し、変換リクエストにすぐ応答できる状態を目指す。
        paths = ensure_milo_ready(require_colmap=not args.allow_missing_colmap)
    except MiloConfigurationError as exc:
        print(f"環境チェックに失敗しました: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"環境セットアップに失敗しました: {exc}")
        return 1

    print(f"MILoルート: {paths.root_dir}")
    print(f"MILoスクリプト: {paths.script_dir}")
    print(f"MILo用Python: {paths.python_exe}")
    print("環境チェックが完了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
