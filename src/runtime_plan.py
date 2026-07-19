import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePlan:
    torch_cuda: str
    torch_version: str
    torchvision_version: str
    torchaudio_version: str
    cuda_version: str
    cuda_winget_version: str
    cuda_installer_file: str
    cuda_installer_url: str


RUNTIME_PLANS = {
    "cu128": RuntimePlan(
        torch_cuda="cu128",
        torch_version="2.11.0",
        torchvision_version="0.26.0",
        torchaudio_version="2.11.0",
        cuda_version="12.8",
        cuda_winget_version="12.8.0",
        cuda_installer_file="cuda_12.8.0_571.96_windows.exe",
        cuda_installer_url="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe",
    ),
    "cu121": RuntimePlan(
        torch_cuda="cu121",
        torch_version="2.3.1",
        torchvision_version="0.18.1",
        torchaudio_version="2.3.1",
        cuda_version="12.1",
        cuda_winget_version="12.1.0",
        cuda_installer_file="cuda_12.1.1_531.14_windows.exe",
        cuda_installer_url="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_531.14_windows.exe",
    ),
    "cu118": RuntimePlan(
        torch_cuda="cu118",
        torch_version="2.3.1",
        torchvision_version="0.18.1",
        torchaudio_version="2.3.1",
        cuda_version="11.8",
        cuda_winget_version="11.8.0",
        cuda_installer_file="cuda_11.8.0_522.06_windows.exe",
        cuda_installer_url="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bat", action="store_true", help="Write SET commands for start.bat.")
    parser.add_argument("--verify-installed", action="store_true", help="Verify installed torch matches the selected runtime.")
    args = parser.parse_args()

    try:
        gpu_name, compute_cap = detect_primary_gpu()
        plan = choose_runtime_plan(gpu_name, compute_cap)
        validate_override(plan, compute_cap)
        if args.verify_installed:
            verify_installed_torch(plan, compute_cap)
    except RuntimeError as exc:
        if args.bat:
            print(f"echo Runtime compatibility check failed: {exc}")
            print("exit /b 1")
        else:
            print(f"Runtime compatibility check failed: {exc}")
        return 0 if args.bat else 1

    if args.bat:
        print(f"echo Detected GPU: {gpu_name} compute capability {compute_cap}")
        print(f"echo Selected runtime: PyTorch {plan.torch_version}+{plan.torch_cuda}, CUDA Toolkit {plan.cuda_version}")
        print(f'set "MILO_GPU_NAME={gpu_name}"')
        print(f'set "MILO_GPU_COMPUTE_CAPABILITY={compute_cap}"')
        print(f'set "MILO_TORCH_CUDA={plan.torch_cuda}"')
        print(f'set "MILO_TORCH_VERSION={plan.torch_version}"')
        print(f'set "MILO_TORCHVISION_VERSION={plan.torchvision_version}"')
        print(f'set "MILO_TORCHAUDIO_VERSION={plan.torchaudio_version}"')
        print(f'set "REQUIRED_CUDA_VERSION={plan.cuda_version}"')
        print(f'set "REQUIRED_CUDA_WINGET_VERSION={plan.cuda_winget_version}"')
        print(f'set "REQUIRED_CUDA_INSTALLER_FILE={plan.cuda_installer_file}"')
        print(f'set "REQUIRED_CUDA_INSTALLER_URL={plan.cuda_installer_url}"')
        print(f'set "MILO_CUDA_VERSION={plan.cuda_version}"')
        print(f'set "TORCH_CUDA_ARCH_LIST={compute_cap}"')
        print("exit /b 0")
    else:
        print(f"{gpu_name} sm_{compute_cap.replace('.', '')}: torch {plan.torch_version}+{plan.torch_cuda}, CUDA {plan.cuda_version}")
    return 0


def verify_installed_torch(plan: RuntimePlan, compute_cap: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed.") from exc

    if torch.version.cuda != plan.cuda_version:
        raise RuntimeError(
            f"Installed PyTorch uses CUDA {torch.version.cuda}, but GPU/program plan requires CUDA {plan.cuda_version}."
        )

    expected_arch = f"sm_{compute_cap.replace('.', '')}"
    supported_arches = set(torch.cuda.get_arch_list())
    if expected_arch not in supported_arches:
        supported = ", ".join(sorted(supported_arches)) or "none"
        raise RuntimeError(
            f"Installed PyTorch does not support GPU architecture {expected_arch}. "
            f"Supported architectures: {supported}."
        )


def detect_primary_gpu() -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found. Install an NVIDIA driver before running MILo.") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed. {detail}")

    rows = list(csv.reader(completed.stdout.splitlines()))
    if not rows:
        raise RuntimeError("No NVIDIA GPU was reported by nvidia-smi.")

    name = rows[0][0].strip()
    compute_cap = rows[0][1].strip()
    if not name or not compute_cap:
        raise RuntimeError("Could not read GPU name and compute capability from nvidia-smi.")
    return name, compute_cap


def choose_runtime_plan(_gpu_name: str, compute_cap: str) -> RuntimePlan:
    major = parse_compute_major(compute_cap)
    if major >= 12:
        return RUNTIME_PLANS["cu128"]
    if major >= 9:
        return RUNTIME_PLANS["cu121"]
    if major >= 5:
        return RUNTIME_PLANS["cu118"]
    raise RuntimeError(f"GPU compute capability {compute_cap} is too old for this MILo setup.")


def validate_override(plan: RuntimePlan, compute_cap: str) -> None:
    requested = os.environ.get("MILO_TORCH_CUDA", "").strip().lower()
    if not requested:
        return
    if requested not in RUNTIME_PLANS:
        raise RuntimeError(f"Unsupported MILO_TORCH_CUDA={requested}. Use cu128, cu121, or cu118.")
    requested_plan = RUNTIME_PLANS[requested]
    required_plan = choose_runtime_plan("", compute_cap)
    if requested_plan.torch_cuda != required_plan.torch_cuda:
        raise RuntimeError(
            f"MILO_TORCH_CUDA={requested} is not compatible with GPU compute capability {compute_cap}. "
            f"Use {required_plan.torch_cuda}."
        )


def parse_compute_major(compute_cap: str) -> int:
    try:
        return int(compute_cap.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid GPU compute capability: {compute_cap}") from exc


if __name__ == "__main__":
    sys.exit(main())
