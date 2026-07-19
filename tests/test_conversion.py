import asyncio
import contextlib
import io

import cv2
import numpy as np
import pytest
import trimesh

import src.main as main


def create_sample_video(video_path):
    """変換テスト用の短いサンプル動画を作成する。"""
    # 明度が異なるフレームを入れて、フレーム抽出や動画読み込みが確認できるようにする。
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    for value in [0, 128, 255, 64]:
        frame = np.full((32, 32, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def create_sample_mesh(mesh_path):
    """USD/OBJエクスポートテスト用の小さなPLYメッシュを作成する。"""
    # icosphere は三角形メッシュなので、アプリのUSD書き出し前提と相性がよい。
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh.export(mesh_path, file_type="ply")


def test_extract_video_frames_creates_colmap_input_images(tmp_path):
    """動画からCOLMAP入力用のJPGフレームが作成されることを確認する。"""
    video_path = tmp_path / "sample.mp4"
    frames_dir = tmp_path / "dataset" / "input"
    create_sample_video(video_path)

    frame_count = main.extract_video_frames(video_path, frames_dir)

    assert frame_count == 4
    assert len(list(frames_dir.glob("*.jpg"))) == 4


def test_export_mesh_creates_obj(tmp_path):
    """PLYメッシュをOBJとして書き出せることを確認する。"""
    source_mesh = tmp_path / "mesh.ply"
    output_path = tmp_path / "mesh.obj"
    create_sample_mesh(source_mesh)

    main.export_mesh(source_mesh, output_path, "obj")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_mesh_creates_openable_usd(tmp_path):
    """PLYメッシュから生成したUSDをOpenUSDで開けることを確認する。"""
    usd = pytest.importorskip("pxr.Usd")
    source_mesh = tmp_path / "mesh.ply"
    output_path = tmp_path / "mesh.usda"
    create_sample_mesh(source_mesh)

    main.export_mesh(source_mesh, output_path, "usda")
    stage = usd.Stage.Open(str(output_path))

    assert stage is not None
    assert str(stage.GetDefaultPrim().GetPath()) == "/Model"


def test_convert_upload_reports_missing_milo(tmp_path, monkeypatch):
    """MILo未設定かつ自動インストール無効時に、分かる例外で止まることを確認する。"""
    video_path = tmp_path / "sample.mp4"
    create_sample_video(video_path)
    monkeypatch.delenv("MILO_ROOT", raising=False)
    monkeypatch.setenv("MILO_AUTO_INSTALL", "0")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "DEFAULT_MILO_ROOT", tmp_path / ".milo" / "MILo")

    upload = main.UploadFile(filename="sample.mp4", file=io.BytesIO(video_path.read_bytes()))

    with pytest.raises(main.MiloConfigurationError):
        asyncio.run(main.convert_upload_with_milo(upload, "obj", "outdoor"))

    assert not list(tmp_path.glob("sample-*.mp4"))
    assert not list(tmp_path.glob("sample-*.obj"))


def test_resolve_or_install_milo_clones_when_missing(tmp_path, monkeypatch):
    """MILoが未配置の場合に、自動clone処理へ進むことを確認する。"""
    milo_root = tmp_path / ".milo" / "MILo"
    script_dir = milo_root / "milo"

    def fake_clone(target_dir):
        """実ネットワークを使わず、clone後のMILo構成だけを疑似的に作る。"""
        assert target_dir == milo_root.resolve()
        script_dir.mkdir(parents=True)
        (script_dir / "train.py").write_text("", encoding="utf-8")
        (script_dir / "mesh_extract_sdf.py").write_text("", encoding="utf-8")

    monkeypatch.delenv("MILO_ROOT", raising=False)
    monkeypatch.setenv("MILO_AUTO_INSTALL", "1")
    monkeypatch.setattr(main, "DEFAULT_MILO_ROOT", milo_root)
    monkeypatch.setattr(main, "clone_milo", fake_clone)

    paths = main.resolve_or_install_milo()

    assert paths.root_dir == milo_root.resolve()
    assert paths.script_dir == script_dir


def test_normalize_gitmodule_urls_rewrites_ssh_github_urls(tmp_path):
    """SSH形式のsubmodule URLがHTTPS形式に書き換わることを確認する。"""
    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text(
        """
[submodule "submodules/Depth-Anything-V2"]
    path = submodules/Depth-Anything-V2
    url = git@github.com:DepthAnything/Depth-Anything-V2.git
[submodule "submodules/nvdiffrast"]
    path = submodules/nvdiffrast
    url = git@github.com:NVlabs/nvdiffrast.git
""",
        encoding="utf-8",
    )

    main.normalize_gitmodule_urls(tmp_path)

    text = gitmodules.read_text(encoding="utf-8")
    assert "https://github.com/DepthAnything/Depth-Anything-V2.git" in text
    assert "https://github.com/NVlabs/nvdiffrast.git" in text
    assert "git@github.com" not in text


def test_ensure_milo_submodules_syncs_and_updates(tmp_path, monkeypatch):
    """既存MILo配置に対してsubmodule同期と取得が実行されることを確認する。"""
    calls = []
    milo_root = tmp_path / "MILo"
    milo_root.mkdir()
    paths = main.MiloPaths(milo_root, milo_root, "python")

    def fake_run_command(command, cwd, step_name):
        """実際のgitは動かさず、呼び出されたsubmoduleコマンドだけ記録する。"""
        calls.append((command, cwd, step_name))

    monkeypatch.setattr(main.shutil, "which", lambda name: "git.exe" if name == "git" else None)
    monkeypatch.setattr(main, "run_command", fake_run_command)

    main.ensure_milo_submodules(paths)

    assert calls[0][0] == ["git.exe", "submodule", "sync", "--recursive"]
    assert calls[1][0] == ["git.exe", "submodule", "update", "--init", "--recursive"]


def test_ensure_milo_ready_reports_missing_colmap(tmp_path, monkeypatch):
    """COLMAP自動インストール無効時に、COLMAP不足として失敗することを確認する。"""
    milo_root = tmp_path / "MILo"
    script_dir = milo_root / "milo"
    script_dir.mkdir(parents=True)
    (script_dir / "train.py").write_text("", encoding="utf-8")
    (script_dir / "mesh_extract_sdf.py").write_text("", encoding="utf-8")
    (milo_root / ".milo_install_complete").write_text("ok\n", encoding="utf-8")

    monkeypatch.setenv("MILO_ROOT", str(milo_root))
    monkeypatch.setenv("COLMAP_AUTO_INSTALL", "0")
    monkeypatch.setattr(main.shutil, "which", lambda name: None)
    monkeypatch.setattr(main, "find_local_colmap_command", lambda: None)
    monkeypatch.setattr(main, "get_python_version", lambda python_exe: "3.11")
    monkeypatch.setattr(main, "find_missing_milo_python_modules", lambda python_exe: [])
    monkeypatch.setattr(main, "ensure_milo_submodules", lambda paths: None)

    with pytest.raises(main.MiloConfigurationError, match="COLMAP"):
        main.ensure_milo_ready()


def test_ensure_colmap_available_installs_with_winget(monkeypatch):
    """Windows環境ではCOLMAP不足時にwingetインストールを試すことを確認する。"""
    calls = []

    def fake_which(name):
        """テスト内だけでwingetとcolmapの検出結果を切り替える。"""
        if name == "winget":
            return "winget.exe"
        if name == "colmap" and calls:
            return "colmap.exe"
        return None

    def fake_run_command(command, cwd, step_name):
        """実際にはwingetを実行せず、呼び出し内容だけ記録する。"""
        calls.append((command, cwd, step_name))

    monkeypatch.setattr(main.os, "name", "nt")
    monkeypatch.setenv("COLMAP_AUTO_INSTALL", "1")
    monkeypatch.setattr(main.shutil, "which", fake_which)
    monkeypatch.setattr(main, "find_local_colmap_command", lambda: None)
    monkeypatch.setattr(main, "run_command", fake_run_command)

    main.ensure_colmap_available()

    assert calls
    assert calls[0][0][:4] == ["winget.exe", "install", "--id", "COLMAP.COLMAP"]


def test_install_milo_dependencies_rechecks_stale_marker(tmp_path, monkeypatch):
    """Existing install markers are ignored when MILo runtime imports are missing."""
    milo_root = tmp_path / "MILo"
    milo_root.mkdir()
    (milo_root / "install.py").write_text("", encoding="utf-8")
    (milo_root / "requirements.txt").write_text("PyYAML\n", encoding="utf-8")
    (milo_root / ".milo_install_complete").write_text("ok\n", encoding="utf-8")
    paths = main.MiloPaths(milo_root, milo_root, "python")
    checks = [["yaml"], ["torch"], ["torch"]]
    calls = []

    def fake_find_missing(_python_exe):
        return checks.pop(0)

    def fake_run_command(command, cwd, step_name):
        calls.append((command, cwd, step_name))

    monkeypatch.setattr(main, "find_missing_milo_python_modules", fake_find_missing)
    monkeypatch.setattr(main, "get_python_version", lambda python_exe: "3.11")
    monkeypatch.setattr(main, "ensure_milo_build_environment", lambda python_exe: None)
    monkeypatch.setattr(main, "run_command", fake_run_command)

    with pytest.raises(main.MiloConfigurationError, match="torch"):
        main.install_milo_dependencies(paths)

    assert calls[0][0][:4] == ["python", "-m", "pip", "install"]
    assert all(call[0] != ["python", str(milo_root / "install.py")] for call in calls)


def test_install_milo_dependencies_rejects_python_313(tmp_path, monkeypatch):
    """MILo dependency setup fails early on Python versions unsupported by pinned packages."""
    milo_root = tmp_path / "MILo"
    milo_root.mkdir()
    paths = main.MiloPaths(milo_root, milo_root, "python")

    monkeypatch.setattr(main, "get_python_version", lambda python_exe: "3.13")

    with pytest.raises(main.MiloConfigurationError, match="Python 3.13"):
        main.install_milo_dependencies(paths)


def test_ensure_colmap_available_falls_back_to_release_zip(monkeypatch):
    """wingetでCOLMAPが見つからない場合に、公式zip取得へフォールバックすることを確認する。"""
    actions = []

    def fake_which(name):
        """COLMAPもwingetもPATH上には無い状態を再現する。"""
        return None

    def fake_install_with_winget():
        """winget側でパッケージが見つからなかった状況を再現する。"""
        raise RuntimeError("No package found matching input criteria.")

    def fake_install_from_zip():
        """実際の大きなzipは取らず、フォールバック到達だけ記録する。"""
        actions.append("zip")

    monkeypatch.setattr(main.os, "name", "nt")
    monkeypatch.setenv("COLMAP_AUTO_INSTALL", "1")
    monkeypatch.setattr(main.shutil, "which", fake_which)
    monkeypatch.setattr(main, "install_colmap_with_winget", fake_install_with_winget)
    monkeypatch.setattr(main, "install_colmap_from_release_zip", fake_install_from_zip)
    monkeypatch.setattr(main, "find_local_colmap_command", lambda: None)

    with pytest.raises(main.MiloConfigurationError, match="COLMAP"):
        main.ensure_colmap_available()

    assert actions == ["zip"]


def test_enqueue_conversion_processes_jobs_one_by_one(tmp_path, monkeypatch):
    """複数の変換依頼が同時に来ても、キューで1件ずつ順番に処理されることを確認する。"""

    async def run_queue_test():
        """テスト専用キューと疑似変換処理で、実MILoを使わず順序だけ検証する。"""
        order = []

        async def fake_convert_file_bytes_with_milo(filename, content, output_format, scene_type):
            """重いMILo処理の代わりに、処理開始順を記録して短く待つ。"""
            order.append(filename)
            await asyncio.sleep(0.01)
            return tmp_path / f"{filename}.{output_format}", 1

        # グローバルキューはアプリ本体用なので、テスト中だけ空のキューへ差し替える。
        monkeypatch.setattr(main, "CONVERSION_QUEUE", asyncio.Queue())
        monkeypatch.setattr(main, "CONVERSION_WORKER_TASK", None)
        monkeypatch.setattr(main, "convert_file_bytes_with_milo", fake_convert_file_bytes_with_milo)

        main.start_conversion_worker()
        upload_a = main.UploadFile(filename="a.mp4", file=io.BytesIO(b"a"))
        upload_b = main.UploadFile(filename="b.mp4", file=io.BytesIO(b"b"))

        await asyncio.gather(
            main.enqueue_conversion(upload_a, "obj", "outdoor"),
            main.enqueue_conversion(upload_b, "obj", "outdoor"),
        )

        assert order == ["a.mp4", "b.mp4"]

        # ワーカーは常駐タスクなので、テスト終了時に明示的に止める。
        main.CONVERSION_WORKER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await main.CONVERSION_WORKER_TASK

    asyncio.run(run_queue_test())
