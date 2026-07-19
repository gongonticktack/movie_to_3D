import asyncio
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple
from zipfile import ZipFile

import cv2
import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# プロジェクト内で使う主要パス。外部へ動画を残さないため、最終成果物だけ OUTPUT_DIR に保存する。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MILO_ROOT = PROJECT_ROOT / ".milo" / "MILo"
DEFAULT_COLMAP_ROOT = PROJECT_ROOT / ".tools" / "COLMAP"
MILO_REPOSITORY_URL = "https://github.com/Anttwo/MILo.git"
COLMAP_WINDOWS_ZIP_URL = "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-nocuda.zip"
SUPPORTED_OUTPUT_FORMATS = {"usda", "obj"}
MILO_REQUIRED_PYTHON_MODULES = {
    "yaml": "PyYAML",
    "torch": "torch",
    "tqdm": "tqdm",
    "plyfile": "plyfile",
    "skimage": "scikit-image",
    "open3d": "open3d",
    "matplotlib": "matplotlib",
    "fused_ssim": "fused-ssim",
    "diff_gaussian_rasterization_ms": "diff-gaussian-rasterization_ms",
    "diff_gaussian_rasterization": "diff-gaussian-rasterization",
    "diff_gaussian_rasterization_gof": "diff-gaussian-rasterization_gof",
    "simple_knn": "simple-knn",
    "nvdiffrast": "nvdiffrast",
    "tetranerf": "tetra-nerf",
}
MILO_LOCAL_PIP_PACKAGES = [
    "submodules/diff-gaussian-rasterization_ms",
    "submodules/diff-gaussian-rasterization",
    "submodules/diff-gaussian-rasterization_gof",
    "submodules/simple-knn",
    "submodules/fused-ssim",
    "submodules/tetra_triangulation",
    "submodules/nvdiffrast",
]

OUTPUT_DIR.mkdir(exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SRC_DIR), str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """アプリの起動時と終了時に、変換キューワーカーの開始と停止を管理する。"""
    start_conversion_worker()
    try:
        yield
    finally:
        # 開発サーバーの再起動時などに、常駐ワーカーを残さないよう明示的に止める。
        global CONVERSION_WORKER_TASK
        if CONVERSION_WORKER_TASK is not None:
            CONVERSION_WORKER_TASK.cancel()
            with suppress(asyncio.CancelledError):
                await CONVERSION_WORKER_TASK
            CONVERSION_WORKER_TASK = None


# FastAPI アプリ本体。画面表示、アップロード受付、ダウンロード配信を担当する。
app = FastAPI(title="MILo Video to 3D Web App", lifespan=lifespan)


class ConversionResponse(BaseModel):
    """変換結果をAPIレスポンスとして返す場合のデータ構造。"""

    output_path: str
    frame_count: int
    output_format: str


class MiloPaths(NamedTuple):
    """MILo 実行に必要なパス一式をまとめる構造。"""

    root_dir: Path
    script_dir: Path
    python_exe: str


class MiloConfigurationError(RuntimeError):
    """MILo や外部コマンドの準備不足を表す例外。"""


@dataclass
class ConversionJob:
    """キューに積む1件分の変換ジョブを表す。"""

    filename: str
    content: bytes
    output_format: str
    scene_type: str
    future: asyncio.Future


# MILo はGPUや外部プロセスを強く使うため、同時実行せず1本のキューで順番に処理する。
CONVERSION_QUEUE: asyncio.Queue[ConversionJob] = asyncio.Queue()
CONVERSION_WORKER_TASK: asyncio.Task | None = None


def start_conversion_worker() -> None:
    """変換キュー用ワーカーが未起動なら開始する。"""
    global CONVERSION_WORKER_TASK

    # テストやリロード時に二重起動しないよう、既存タスクが生きている場合はそのまま使う。
    if CONVERSION_WORKER_TASK is not None and not CONVERSION_WORKER_TASK.done():
        return
    CONVERSION_WORKER_TASK = asyncio.create_task(conversion_worker())


async def conversion_worker() -> None:
    """変換キューからジョブを取り出し、1件ずつ順番にMILo変換を実行する。"""
    while True:
        job = await CONVERSION_QUEUE.get()
        try:
            # 実際の変換処理はここでだけ呼ぶため、複数リクエストが来ても同時実行されない。
            result = await convert_file_bytes_with_milo(
                job.filename,
                job.content,
                job.output_format,
                job.scene_type,
            )
            if not job.future.done():
                job.future.set_result(result)
        except Exception as exc:
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            CONVERSION_QUEUE.task_done()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """動画アップロード用のHTMLフォームを返す。"""
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>MILo Video to 3D</title>
            <style>
                body { font-family: Arial, sans-serif; background: #07111f; color: #f5f7fb; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
                .card { width: min(760px, 90vw); background: rgba(9, 22, 40, 0.94); border: 1px solid #2b3d5b; border-radius: 8px; padding: 32px; box-shadow: 0 20px 50px rgba(0,0,0,0.35); }
                label { display: block; margin: 18px 0 8px; color: #c7d2fe; }
                input, select, button { font-size: 16px; padding: 10px 14px; border-radius: 8px; border: 1px solid #35506e; max-width: 100%; }
                select { background: #0d1b2f; color: #f5f7fb; }
                button { margin-top: 22px; background: #22d3ee; color: #06111d; cursor: pointer; font-weight: 700; }
                .note { color: #a7b4ca; line-height: 1.6; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>MILo Video to 3D</h1>
                <p class="note">動画からフレームを抽出し、MILoで3Dメッシュを再構成して出力します。不足しているMILo本体は自動セットアップします。</p>
                <form action="/convert" method="post" enctype="multipart/form-data">
                    <label for="file">動画ファイル</label>
                    <input id="file" type="file" name="file" accept="video/*" required>

                    <label for="output_format">出力形式</label>
                    <select id="output_format" name="output_format">
                        <option value="usda">USD ASCII (.usda)</option>
                        <option value="obj">OBJ (.obj)</option>
                    </select>

                    <label for="scene_type">シーン種別</label>
                    <select id="scene_type" name="scene_type">
                        <option value="outdoor">outdoor</option>
                        <option value="indoor">indoor</option>
                    </select>

                    <button type="submit">3Dモデル生成</button>
                </form>
            </div>
        </body>
        </html>
        """
    )


@app.post("/convert", response_class=HTMLResponse)
async def convert_video(
    file: UploadFile = File(...),
    output_format: str = Form("usda"),
    scene_type: str = Form("outdoor"),
) -> HTMLResponse:
    """アップロード動画をMILoで3Dモデル化し、ダウンロードリンクを返す。"""
    try:
        result_path, frame_count = await enqueue_conversion(file, output_format, scene_type)
    except (MiloConfigurationError, RuntimeError, ValueError) as exc:
        return render_error(str(exc))

    # ファイル名はHTMLへ埋め込むため、念のためエスケープしてからリンク化する。
    link = html.escape(result_path.name)
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="ja">
        <body style="font-family:Arial;background:#07111f;color:#f5f7fb;padding:32px">
            <h1>生成完了</h1>
            <p>使用フレーム数: {frame_count}</p>
            <p><a href="/download/{link}" style="color:#22d3ee">3Dモデルをダウンロード</a></p>
        </body></html>
        """
    )


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    """output フォルダ内の生成済み3Dモデルをダウンロード配信する。"""
    target = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()

    # パストラバーサル対策として、解決後のパスが output 配下にあることを必ず確認する。
    if output_root not in target.parents or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target, filename=target.name)


async def convert_upload_with_milo(
    file: UploadFile,
    output_format: str,
    scene_type: str,
) -> tuple[Path, int]:
    """テストや内部呼び出し用に、UploadFileを直接MILo変換する。"""
    return await convert_file_bytes_with_milo(
        file.filename or "upload.mp4",
        await file.read(),
        output_format,
        scene_type,
    )


async def enqueue_conversion(
    file: UploadFile,
    output_format: str,
    scene_type: str,
) -> tuple[Path, int]:
    """アップロード内容をジョブ化し、変換キューで順番が来るまで待機する。"""
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    # UploadFile はリクエスト終了後に閉じられる可能性があるため、キュー投入前にbytesへ読み切る。
    content = await file.read()
    job = ConversionJob(
        filename=file.filename or "upload.mp4",
        content=content,
        output_format=output_format,
        scene_type=scene_type,
        future=future,
    )

    # put は非同期なので、同時に複数リクエストが来ても投入順がキューに保持される。
    await CONVERSION_QUEUE.put(job)
    return await future


async def convert_file_bytes_with_milo(
    filename: str,
    content: bytes,
    output_format: str,
    scene_type: str,
) -> tuple[Path, int]:
    """アップロードファイルを一時領域に置き、MILo変換と最終形式への書き出しを実行する。"""
    output_format = normalize_output_format(output_format)
    scene_type = normalize_scene_type(scene_type)
    source_name = Path(filename or "upload.mp4")
    safe_stem = sanitize_filename_part(source_name.stem or "upload")
    source_suffix = sanitize_suffix(source_name.suffix or ".mp4")
    asset_id = uuid.uuid4().hex[:8]

    # 一時ディレクトリ内だけに入力動画を保存することで、変換後に元動画が残らないようにする。
    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        workspace = Path(tmpdir)
        video_path = workspace / f"{safe_stem}{source_suffix}"
        with video_path.open("wb") as fh:
            fh.write(content)

        # MILo は動画ではなく COLMAP 形式の画像群を入力にするため、まずフレーム画像を作る。
        dataset_dir = workspace / "dataset"
        frame_count = await run_in_threadpool(extract_video_frames, video_path, dataset_dir / "input")

        # MILo の学習とメッシュ抽出は重い処理なので、イベントループを塞がないようスレッドへ逃がす。
        milo_output_dir = workspace / "milo_output"
        ply_path = await run_in_threadpool(run_milo_reconstruction, dataset_dir, milo_output_dir, scene_type)

        # MILo が出力した PLY を、ユーザーが選んだ最終形式へ変換して output に保存する。
        final_path = OUTPUT_DIR / f"{safe_stem}-{asset_id}.{output_format}"
        await run_in_threadpool(export_mesh, ply_path, final_path, output_format)
        return final_path, frame_count


def extract_video_frames(video_path: Path, frames_dir: Path, max_frames: int = 96) -> int:
    """動画から等間隔でフレームを抽出し、COLMAP が読む input フォルダへ JPG 保存する。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("動画を開けませんでした。対応している動画ファイルか確認してください。")

    # 長い動画をそのまま全フレーム処理するとMILoが非常に重くなるため、最大枚数に間引く。
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(total_frames // max_frames, 1) if total_frames else 1
    saved = 0
    frame_index = 0

    try:
        while saved < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frame_path = frames_dir / f"{saved:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved += 1
            frame_index += 1
    finally:
        cap.release()

    # 3D再構成には複数視点が必要なので、最低限のフレーム数を満たさない入力は早めに止める。
    if saved < 3:
        raise ValueError("MILoで再構成するには、少なくとも3フレーム以上読み取れる動画が必要です。")
    return saved


def run_milo_reconstruction(dataset_dir: Path, output_dir: Path, scene_type: str) -> Path:
    """MILo の前処理、学習、メッシュ抽出を順番に実行し、生成された PLY を返す。"""
    milo = ensure_milo_ready()
    output_dir.mkdir(parents=True, exist_ok=True)

    # convert.py は dataset/input の画像群から COLMAP 用のカメラ情報を作る。
    run_command(
        [milo.python_exe, str(milo.script_dir / "convert.py"), "-s", str(dataset_dir), "--no_gpu"],
        cwd=milo.script_dir,
        step_name="COLMAPデータセット作成",
    )

    # train.py は MILo のメイン学習処理。scene_type は indoor/outdoor の重要度設定に使う。
    run_command(
        [
            milo.python_exe,
            "train.py",
            "-s",
            str(dataset_dir),
            "-m",
            str(output_dir),
            "--imp_metric",
            scene_type,
            "--rasterizer",
            "radegs",
        ],
        cwd=milo.script_dir,
        step_name="MILo学習",
    )

    # 学習済み出力から PLY メッシュを抽出し、後段で USD/OBJ に変換する。
    run_command(
        [
            milo.python_exe,
            "mesh_extract_sdf.py",
            "-s",
            str(dataset_dir),
            "-m",
            str(output_dir),
            "--rasterizer",
            "radegs",
        ],
        cwd=milo.script_dir,
        step_name="MILoメッシュ抽出",
    )
    return find_latest_mesh(output_dir)


def ensure_milo_ready(require_colmap: bool = True) -> MiloPaths:
    """MILo 本体、MILo 依存関係、COLMAP の準備状態を確認する。"""
    paths = resolve_or_install_milo()
    ensure_milo_submodules(paths)
    install_milo_dependencies(paths)
    if require_colmap:
        ensure_colmap_available()
    return paths


def resolve_or_install_milo() -> MiloPaths:
    """MILo の場所を解決し、未取得なら自動インストール設定に従って clone する。"""
    configured_root = os.environ.get("MILO_ROOT")
    python_exe = os.environ.get("MILO_PYTHON", sys.executable)

    # ユーザーが MILO_ROOT を指定している場合は、その環境を最優先で使う。
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        script_dir = find_milo_script_dir(root)
        if script_dir is None:
            raise MiloConfigurationError(
                f"MILO_ROOT={root} に train.py と mesh_extract_sdf.py が見つかりません。MILoのパスを確認してください。"
            )
        return MiloPaths(root, script_dir, python_exe)

    # 未指定の場合はプロジェクト配下の .milo/MILo を標準の配置場所として扱う。
    root = DEFAULT_MILO_ROOT.resolve()
    script_dir = find_milo_script_dir(root)
    if script_dir is None:
        if not auto_install_enabled():
            raise MiloConfigurationError(
                "MILoが未設定です。MILO_ROOTを指定するか、MILO_AUTO_INSTALL=1で自動インストールを有効にしてください。"
            )
        clone_milo(root)
        script_dir = find_milo_script_dir(root)

    if script_dir is None:
        raise MiloConfigurationError(f"MILoを {root} に取得しましたが、必要なスクリプトが見つかりません。")

    return MiloPaths(root, script_dir, python_exe)


def find_milo_script_dir(root: Path) -> Path | None:
    """MILo 実行スクリプトが置かれているディレクトリを探す。"""
    # MILo リポジトリは root 直下または root/milo 配下に実行スクリプトを持つ想定で探す。
    candidates = [root, root / "milo"]
    for candidate in candidates:
        if (candidate / "train.py").exists() and (candidate / "mesh_extract_sdf.py").exists():
            return candidate
    return None


def clone_milo(target_dir: Path) -> None:
    """公式 MILo リポジトリをプロジェクト配下へ clone する。"""
    git_exe = shutil.which("git")
    if git_exe is None:
        raise MiloConfigurationError("MILoの自動取得にはgitが必要です。gitをインストールしてください。")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [git_exe, "clone", MILO_REPOSITORY_URL, str(target_dir)],
        cwd=PROJECT_ROOT,
        step_name="MILoリポジトリ取得",
    )


def ensure_milo_submodules(paths: MiloPaths) -> None:
    """MILo の submodule URL を HTTPS に直し、不足している submodule を取得する。"""
    git_exe = shutil.which("git")
    if git_exe is None:
        raise MiloConfigurationError("MILoのsubmodule取得にはgitが必要です。gitをインストールしてください。")

    # MILo の .gitmodules は SSH URL を含むことがあり、通常のWindows環境では鍵未設定で失敗しやすい。
    normalize_gitmodule_urls(paths.root_dir)

    # 既に途中まで clone された環境でも、ここで submodule を再同期して修復できるようにする。
    run_command(
        [git_exe, "submodule", "sync", "--recursive"],
        cwd=paths.root_dir,
        step_name="MILo submodule設定同期",
    )
    run_command(
        [git_exe, "submodule", "update", "--init", "--recursive"],
        cwd=paths.root_dir,
        step_name="MILo submodule取得",
    )


def normalize_gitmodule_urls(repo_dir: Path) -> None:
    """git@github.com形式のsubmodule URLをHTTPS形式へ書き換える。"""
    gitmodules_path = repo_dir / ".gitmodules"
    if not gitmodules_path.exists():
        return

    text = gitmodules_path.read_text(encoding="utf-8")
    normalized = re.sub(r"git@github\.com:([^ \n]+?)(?:\.git)?$", r"https://github.com/\1.git", text, flags=re.MULTILINE)
    if normalized != text:
        gitmodules_path.write_text(normalized, encoding="utf-8")


def install_milo_dependencies(paths: MiloPaths) -> None:
    """Install MILo dependencies without letting upstream install.py overwrite torch."""
    if os.environ.get("MILO_SKIP_INSTALL", "").lower() in {"1", "true", "yes", "on"}:
        return

    python_version = get_python_version(paths.python_exe)
    if python_version.startswith("3.13"):
        raise MiloConfigurationError(
            "MILo cannot be installed into Python 3.13 because its pinned dependencies, "
            "including native CUDA extensions, do not support this environment. "
            "Install Python 3.10/3.11, delete .venv, then run start.bat again. "
            "Alternatively set MILO_PYTHON to a compatible conda/python.exe."
        )

    # 成功済みマーカーがあれば、毎回の起動で重いインストールを繰り返さない。
    marker = paths.root_dir / ".milo_install_complete"
    missing_modules = find_missing_milo_python_modules(paths.python_exe)
    if marker.exists() and not missing_modules:
        return

    ensure_milo_build_environment(paths.python_exe)
    install_milo_python_requirements(paths)
    missing_modules = find_missing_milo_python_modules(paths.python_exe)
    if not missing_modules:
        marker.write_text("ok\n", encoding="utf-8")
        return
    missing_packages = ", ".join(MILO_REQUIRED_PYTHON_MODULES[module] for module in missing_modules)
    hint = ""
    if python_version.startswith("3.13"):
        hint = "\nUse Python 3.10/3.11 for MILo's native extension builds."
    raise MiloConfigurationError(f"MILo Python dependencies are missing after local install: {missing_packages}.{hint}")


def install_milo_python_requirements(paths: MiloPaths) -> None:
    """Install MILo's Python requirements and local extension packages."""
    requirements_path = paths.root_dir / "requirements.txt"
    if requirements_path.exists():
        run_command(
            [paths.python_exe, "-m", "pip", "install", "-r", str(requirements_path)],
            cwd=paths.root_dir,
            step_name="MILo Python requirements install",
        )
    for package_path in MILO_LOCAL_PIP_PACKAGES:
        if (paths.root_dir / package_path).exists():
            run_command(
                [paths.python_exe, "-m", "pip", "install", "--no-build-isolation", package_path],
                cwd=paths.root_dir,
                step_name=f"MILo local package install: {package_path}",
            )
    run_command(
        [paths.python_exe, "-m", "pip", "install", "PyYAML", "matplotlib"],
        cwd=paths.root_dir,
        step_name="MILo supplemental Python requirements install",
    )


def ensure_milo_build_environment(python_exe: str) -> None:
    """Validate tools required to build MILo's CUDA extension modules."""
    torch_cuda = get_torch_cuda_version(python_exe)
    if torch_cuda in {"", "None", "unknown"}:
        raise MiloConfigurationError(
            "MILo requires CUDA-enabled PyTorch to build and run its rasterizers, but the current "
            "environment has CPU-only PyTorch. Install a CUDA build of torch/torchvision/torchaudio "
            "matching your CUDA Toolkit, then run start.bat again."
        )

    cuda_home = find_cuda_home(torch_cuda)
    if cuda_home is None:
        raise MiloConfigurationError(
            f"CUDA Toolkit {torch_cuda} was not found. Install NVIDIA CUDA Toolkit {torch_cuda}, "
            "make sure its nvcc is on PATH, or set CUDA_HOME to that CUDA install directory."
        )

    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    cuda_bin = cuda_home / "bin"
    if cuda_bin.exists():
        os.environ["PATH"] = os.pathsep.join([str(cuda_bin), os.environ.get("PATH", "")])

    if os.name == "nt" and shutil.which("cl") is None:
        raise MiloConfigurationError(
            "Microsoft C++ Build Tools were not found. Install Visual Studio Build Tools with "
            "'Desktop development with C++', open a Developer Command Prompt, then run start.bat again."
        )

    arch_list = get_torch_cuda_arch_list(python_exe)
    if arch_list:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)


def get_torch_cuda_version(python_exe: str) -> str:
    """Return torch.version.cuda for the selected Python environment."""
    try:
        completed = subprocess.run(
            [python_exe, "-c", "import torch; print(torch.version.cuda)"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def get_torch_cuda_arch_list(python_exe: str) -> str:
    """Return the active GPU architecture for PyTorch extension builds."""
    script = (
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    print('')\n"
        "else:\n"
        "    major, minor = torch.cuda.get_device_capability(0)\n"
        "    print(f'{major}.{minor}')\n"
    )
    try:
        completed = subprocess.run(
            [python_exe, "-c", script],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def find_cuda_home(required_version: str | None = None) -> Path | None:
    """Find a local CUDA Toolkit install directory matching torch.version.cuda."""
    required_name = f"v{required_version}" if required_version else None
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured and (Path(configured) / "bin" / "nvcc.exe").exists():
        if required_name is None or Path(configured).name == required_name:
            return Path(configured)

    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if required_name:
        candidate = cuda_root / required_name
        if (candidate / "bin" / "nvcc.exe").exists():
            return candidate

    if configured and (Path(configured) / "bin" / "nvcc.exe").exists() and required_name is None:
        return Path(configured)

    nvcc = shutil.which("nvcc")
    if nvcc and required_name is None:
        return Path(nvcc).resolve().parent.parent

    if cuda_root.exists() and required_name is None:
        candidates = sorted(cuda_root.glob("v*"), reverse=True)
        for candidate in candidates:
            if (candidate / "bin" / "nvcc.exe").exists():
                return candidate
    return None


def find_missing_milo_python_modules(python_exe: str) -> list[str]:
    """Return MILo runtime modules that cannot be imported by python_exe."""
    script = (
        "import importlib.util\n"
        f"mods = {list(MILO_REQUIRED_PYTHON_MODULES)!r}\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        completed = subprocess.run(
            [python_exe, "-c", script],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return list(MILO_REQUIRED_PYTHON_MODULES)
    if completed.returncode != 0:
        return list(MILO_REQUIRED_PYTHON_MODULES)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def get_python_version(python_exe: str) -> str:
    """Return the major.minor Python version for diagnostics."""
    try:
        completed = subprocess.run(
            [python_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def ensure_colmap_available() -> None:
    """COLMAP が利用可能か確認し、不足時は利用可能な方法で準備する。"""
    if find_colmap_command() is not None:
        return

    # 以前の起動で portable 版を取得済みの場合は、PATHへ追加するだけで使えるようにする。
    local_colmap = find_local_colmap_command()
    if local_colmap is not None:
        add_colmap_to_path(local_colmap)
        return

    # COLMAP は Python パッケージではなくOS側の実行ファイルなので、Windowsでは外部配布物で補う。
    if os.name == "nt" and os.environ.get("COLMAP_AUTO_INSTALL", "1").lower() in {"1", "true", "yes", "on"}:
        install_colmap_for_windows()
        if find_colmap_command() is not None:
            return
        raise MiloConfigurationError(
            "COLMAPをインストールしましたが、このプロセスではまだ見つかりません。"
            "start.batを開き直すか、COLMAPのフォルダをPATHに追加してください。"
        )

    raise MiloConfigurationError(
        "COLMAPが見つかりません。MILoの前処理に必要です。COLMAPをインストールしてPATHに追加してください。"
    )


def find_colmap_command() -> str | None:
    """PATH 上から COLMAP のコマンドを探す。"""
    # 公式Windowsバイナリは COLMAP.bat 経由でCLIを提供するため、colmap と COLMAP.bat の両方を見る。
    return shutil.which("colmap") or shutil.which("COLMAP.bat") or shutil.which("colmap.exe")


def find_local_colmap_command() -> Path | None:
    """プロジェクト配下に取得済みの portable COLMAP コマンドを探す。"""
    if not DEFAULT_COLMAP_ROOT.exists():
        return None

    for pattern in ("COLMAP.bat", "colmap.exe"):
        matches = list(DEFAULT_COLMAP_ROOT.rglob(pattern))
        if matches:
            return matches[0]
    return None


def add_colmap_to_path(command_path: Path) -> None:
    """取得済みCOLMAPを現在のPythonプロセスのPATHへ追加する。"""
    # COLMAP.bat はルート側、colmap.exe は bin 側にあることが多いため、親とbin候補の両方を足す。
    path_entries = [str(command_path.parent)]
    bin_dir = command_path.parent / "bin"
    if bin_dir.exists():
        path_entries.append(str(bin_dir))
    os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])


def install_colmap_for_windows() -> None:
    """Windows向けに winget と公式zipの順で COLMAP インストールを試す。"""
    try:
        install_colmap_with_winget()
    except RuntimeError as exc:
        # winget の公式リポジトリにCOLMAPが無い環境があるため、その場合は公式zipへフォールバックする。
        print(f"wingetでのCOLMAPインストールをスキップします: {exc}")
    except MiloConfigurationError as exc:
        print(f"wingetでのCOLMAPインストールをスキップします: {exc}")

    if find_colmap_command() is not None:
        return

    install_colmap_from_release_zip()


def install_colmap_with_winget() -> None:
    """Windows Package Manager を使って COLMAP をインストールする。"""
    winget_exe = shutil.which("winget")
    if winget_exe is None:
        raise MiloConfigurationError(
            "COLMAPが見つからず、wingetも見つからないため自動インストールできません。"
            "COLMAPを手動でインストールしてPATHに追加してください。"
        )

    run_command(
        [
            winget_exe,
            "install",
            "--id",
            "COLMAP.COLMAP",
            "-e",
            "--source",
            "winget",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
        cwd=PROJECT_ROOT,
        step_name="COLMAPインストール",
    )


def install_colmap_from_release_zip() -> None:
    """COLMAP公式GitHub ReleaseのWindows zipを取得して展開する。"""
    DEFAULT_COLMAP_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = DEFAULT_COLMAP_ROOT / "colmap-x64-windows.zip"

    print("COLMAP公式zipをダウンロードしています...")
    try:
        urllib.request.urlretrieve(COLMAP_WINDOWS_ZIP_URL, archive_path)
    except Exception as exc:
        raise MiloConfigurationError(
            "COLMAP公式zipのダウンロードに失敗しました。ネットワーク接続を確認するか、COLMAPを手動でインストールしてください。"
        ) from exc

    print("COLMAP公式zipを展開しています...")
    with ZipFile(archive_path) as zip_file:
        zip_file.extractall(DEFAULT_COLMAP_ROOT)

    local_colmap = find_local_colmap_command()
    if local_colmap is None:
        raise MiloConfigurationError("COLMAP公式zipを展開しましたが、COLMAP.bat または colmap.exe が見つかりません。")
    add_colmap_to_path(local_colmap)


def auto_install_enabled() -> bool:
    """MILo の自動 clone / install が有効か判定する。"""
    value = os.environ.get("MILO_AUTO_INSTALL", "1").lower()
    return value in {"1", "true", "yes", "on"}


def run_command(command: List[str], cwd: Path, step_name: str) -> None:
    """外部コマンドを実行し、失敗時は画面表示しやすい例外へ変換する。"""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise MiloConfigurationError(f"{step_name}に失敗しました。実行ファイルが見つかりません: {exc}") from exc

    if completed.returncode != 0:
        # 長すぎるログは画面を埋めるため、末尾だけ残して原因が追える量にする。
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 3000:
            detail = detail[-3000:]
        raise RuntimeError(f"{step_name}に失敗しました。\n{detail}")


def find_latest_mesh(output_dir: Path) -> Path:
    """MILo 出力フォルダから最新の PLY メッシュを探す。"""
    meshes = sorted(output_dir.rglob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not meshes:
        raise RuntimeError("MILoの出力フォルダにPLYメッシュが見つかりませんでした。")
    return meshes[0]


def export_mesh(source_mesh: Path, target_path: Path, output_format: str) -> None:
    """MILo の PLY メッシュを、アプリで選択された USD または OBJ へ変換する。"""
    mesh = trimesh.load(source_mesh, force="mesh")
    if mesh.is_empty:
        raise RuntimeError("MILoが空のメッシュを出力しました。")

    # OBJ は trimesh の標準エクスポータに任せ、USD は互換性重視で独自に usda を書く。
    if output_format == "obj":
        mesh.export(target_path, file_type="obj")
    elif output_format == "usda":
        write_mesh_usda(mesh, target_path)
    else:
        raise ValueError("対応している出力形式は USD または OBJ です。")


def write_mesh_usda(mesh: trimesh.Trimesh, target_path: Path) -> None:
    """trimesh のメッシュを最小構成の USD ASCII として書き出す。"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # defaultPrim を付けることで、USDビューアが開いたときにルートモデルを認識しやすくする。
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Model"',
        ")",
        "",
        'def Xform "Model" {',
        '    def Mesh "Mesh" {',
        "        point3f[] points = [",
    ]
    for index, vertex in enumerate(vertices):
        suffix = "," if index < len(vertices) - 1 else ""
        lines.append(f"            ({vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f}){suffix}")
    lines.append("        ]")

    # MILo の抽出メッシュは三角形として扱うため、faceVertexCounts はすべて 3 にする。
    lines.append(f"        int[] faceVertexCounts = [{', '.join(['3'] * len(faces))}]")
    face_indices = [str(int(value)) for face in faces for value in face]
    lines.append(f"        int[] faceVertexIndices = [{', '.join(face_indices)}]")
    lines.append("    }")
    lines.append("}")
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_output_format(value: str) -> str:
    """出力形式の表記ゆれを正規化し、未対応形式を拒否する。"""
    normalized = value.lower().lstrip(".")
    if normalized == "usd":
        normalized = "usda"
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError("対応している出力形式は USD または OBJ です。")
    return normalized


def normalize_scene_type(value: str) -> str:
    """MILo の重要度設定に使うシーン種別を検証する。"""
    normalized = value.lower()
    if normalized not in {"indoor", "outdoor"}:
        raise ValueError("scene_typeは indoor または outdoor を指定してください。")
    return normalized


def sanitize_filename_part(value: str) -> str:
    """ユーザー由来のファイル名から、安全に使えるベース名を作る。"""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "upload"


def sanitize_suffix(value: str) -> str:
    """ユーザー由来の拡張子を、短い英数字拡張子だけに制限する。"""
    if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", value):
        return value
    return ".mp4"


def render_error(message: str) -> HTMLResponse:
    """例外メッセージを安全にHTML化し、エラー画面として返す。"""
    safe_message = html.escape(message).replace("\n", "<br>")
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="ja">
        <body style="font-family:Arial;background:#07111f;color:#f5f7fb;padding:32px">
            <h1>生成できませんでした</h1>
            <p>{safe_message}</p>
            <p><a href="/" style="color:#22d3ee">戻る</a></p>
        </body></html>
        """,
        status_code=500,
    )
