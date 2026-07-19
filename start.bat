@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if "%MOV23D_KEEP_OPEN%"=="" if not "%MOV23D_NO_CMD_K%"=="1" (
    set "MOV23D_KEEP_OPEN=1"
    start "MILo Video to 3D" "%ComSpec%" /k call "%~f0"
    exit /b 0
)

echo Starting MILo Video to 3D app...
echo Press Ctrl+C to stop the server.
echo.

set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8765"
set "SELECTED_PORT="
set /a END_PORT=PORT+20

call :find_python
if errorlevel 1 goto failed

call :require_command git "Git is required to download and update MILo."
if errorlevel 1 goto failed

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto failed
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
set "PATH=%CD%\.venv\Scripts;%PATH%"

set "PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%"
set "MILO_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist ".tmp" mkdir ".tmp"
python -m src.runtime_plan --bat > ".tmp\runtime_plan.bat"
if errorlevel 1 goto failed
call ".tmp\runtime_plan.bat"
if errorlevel 1 goto failed

echo Checking Python packaging tools...
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip setuptools wheel ninja
if errorlevel 1 goto failed

call :ensure_cuda_toolkit
if errorlevel 1 goto failed

call :ensure_cpp_build_tools
if errorlevel 1 goto failed
set "DISTUTILS_USE_SDK=1"

echo Installing missing Python libraries...
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Installing CUDA-enabled PyTorch for MILo...
python -m pip uninstall -y torch torchvision torchaudio
if errorlevel 1 goto failed
python -m pip install --index-url https://download.pytorch.org/whl/%MILO_TORCH_CUDA% --force-reinstall torch==%MILO_TORCH_VERSION% torchvision==%MILO_TORCHVISION_VERSION% torchaudio==%MILO_TORCHAUDIO_VERSION%
if errorlevel 1 goto failed
python -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)"
if errorlevel 1 (
    echo.
    echo CUDA-enabled PyTorch could not be installed. CPU-only PyTorch is still active.
    echo Check MILO_TORCH_CUDA, CUDA Toolkit, and the PyTorch wheel download result.
    goto failed
)
python -m src.runtime_plan --verify-installed
if errorlevel 1 goto failed

echo Checking external runtime tools and MILo environment...
python -m src.preflight
if errorlevel 1 goto failed

for /l %%P in (%PORT%,1,%END_PORT%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %%P -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
    if not errorlevel 1 (
        set "SELECTED_PORT=%%P"
        goto port_found
    )
)

echo.
echo No available port found between %PORT% and %END_PORT%.
echo Close another server or run with a different PORT value, for example:
echo   set PORT=9000
echo   start.bat
goto failed

:port_found
if not "%SELECTED_PORT%"=="%PORT%" (
    echo Port %PORT% is already in use. Using port %SELECTED_PORT% instead.
)
echo Open http://%HOST%:%SELECTED_PORT%/ in your browser.
echo.

python -m uvicorn src.main:app --host %HOST% --port %SELECTED_PORT%
if errorlevel 1 goto failed
goto finished

:find_python
set "PYTHON_CMD="

python -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)

py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    exit /b 0
)

if exist "E:\python\3.11\python.exe" (
    set "PYTHON_CMD=""E:\python\3.11\python.exe"""
    exit /b 0
)

if exist "E:\python\311\python.exe" (
    set "PYTHON_CMD=""E:\python\311\python.exe"""
    exit /b 0
)

echo.
echo Python was not found. Install Python 3.10 or 3.11, then run start.bat again.
exit /b 1

:require_command
where %~1 >nul 2>&1
if not errorlevel 1 exit /b 0

echo.
echo %~1 was not found.
echo %~2
exit /b 1

:ensure_cuda_toolkit
call :detect_cuda_toolkit
if defined CUDA_HOME exit /b 0

where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo CUDA Toolkit was not found, and winget is not available. Trying NVIDIA official installer...
    call :install_cuda_from_nvidia
    if errorlevel 1 (
        echo Install NVIDIA CUDA Toolkit %REQUIRED_CUDA_VERSION% manually, then run start.bat again.
        exit /b 1
    )
    exit /b 0
)

echo CUDA Toolkit %REQUIRED_CUDA_VERSION% was not found. Installing with winget...
winget install --id Nvidia.CUDA -e --version %REQUIRED_CUDA_VERSION% --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo CUDA Toolkit %REQUIRED_CUDA_VERSION% was not available with that exact winget version. Trying %REQUIRED_CUDA_WINGET_VERSION%...
    winget install --id Nvidia.CUDA -e --version %REQUIRED_CUDA_WINGET_VERSION% --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo CUDA Toolkit %REQUIRED_CUDA_VERSION% could not be installed with winget.
        call :install_cuda_from_nvidia
        if errorlevel 1 exit /b 1
    )
)
call :detect_cuda_toolkit
if defined CUDA_HOME exit /b 0

echo CUDA Toolkit %REQUIRED_CUDA_VERSION% was not found after winget versioned install. Trying NVIDIA official installer...
call :install_cuda_from_nvidia
if errorlevel 1 (
    echo CUDA Toolkit %REQUIRED_CUDA_VERSION% could not be installed with NVIDIA official installer.
    echo Trying winget default package as a last resort...
    winget install --id Nvidia.CUDA -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 exit /b 1
)

call :detect_cuda_toolkit
if defined CUDA_HOME exit /b 0

echo CUDA Toolkit %REQUIRED_CUDA_VERSION% was not found after NVIDIA official installer. Trying winget default package...
winget install --id Nvidia.CUDA -e --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1

call :detect_cuda_toolkit
if defined CUDA_HOME exit /b 0

echo.
echo CUDA Toolkit was installed, but CUDA Toolkit %REQUIRED_CUDA_VERSION% was not found in this shell.
echo Install CUDA Toolkit %REQUIRED_CUDA_VERSION% or set CUDA_HOME to that version, then run start.bat again.
exit /b 1

:install_cuda_from_nvidia
if "%REQUIRED_CUDA_INSTALLER_URL%"=="" exit /b 1

set "CUDA_INSTALLER_DIR=%CD%\.tools\cuda-installers"
set "CUDA_INSTALLER=%CUDA_INSTALLER_DIR%\%REQUIRED_CUDA_INSTALLER_FILE%"
if not exist "%CUDA_INSTALLER_DIR%" mkdir "%CUDA_INSTALLER_DIR%"

if not exist "%CUDA_INSTALLER%" (
    echo Downloading CUDA Toolkit %REQUIRED_CUDA_VERSION% installer from NVIDIA...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%REQUIRED_CUDA_INSTALLER_URL%' -OutFile '%CUDA_INSTALLER%'"
    if errorlevel 1 exit /b 1
)

echo Installing CUDA Toolkit %REQUIRED_CUDA_VERSION% from NVIDIA installer...
"%CUDA_INSTALLER%" -s -n
if errorlevel 1 exit /b 1

call :detect_cuda_toolkit
if defined CUDA_HOME exit /b 0
exit /b 1

:detect_cuda_toolkit
set "DETECTED_CUDA_HOME="

if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%REQUIRED_CUDA_VERSION%\bin\nvcc.exe" (
    set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%REQUIRED_CUDA_VERSION%"
    set "CUDA_PATH=%CUDA_HOME%"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    exit /b 0
)

if defined CUDA_HOME call :use_cuda_if_matches "%CUDA_HOME%"
if defined DETECTED_CUDA_HOME (
    set "CUDA_HOME=%DETECTED_CUDA_HOME%"
    set "CUDA_PATH=%DETECTED_CUDA_HOME%"
    set "PATH=%DETECTED_CUDA_HOME%\bin;%PATH%"
    exit /b 0
)

if defined CUDA_PATH call :use_cuda_if_matches "%CUDA_PATH%"
if defined DETECTED_CUDA_HOME (
    set "CUDA_HOME=%DETECTED_CUDA_HOME%"
    set "CUDA_PATH=%DETECTED_CUDA_HOME%"
    set "PATH=%DETECTED_CUDA_HOME%\bin;%PATH%"
    exit /b 0
)

exit /b 0

:use_cuda_if_matches
if /I not "%~nx1"=="v%REQUIRED_CUDA_VERSION%" exit /b 0
if not exist "%~1\bin\nvcc.exe" exit /b 0
set "DETECTED_CUDA_HOME=%~1"
exit /b 0

:set_required_cuda_version
if /I "%MILO_TORCH_CUDA%"=="cu128" (
    set "REQUIRED_CUDA_VERSION=12.8"
    set "REQUIRED_CUDA_WINGET_VERSION=12.8.0"
    set "REQUIRED_CUDA_INSTALLER_FILE=cuda_12.8.0_571.96_windows.exe"
    set "REQUIRED_CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe"
    set "MILO_CUDA_VERSION=12.8"
    if "%MILO_TORCH_VERSION%"=="" set "MILO_TORCH_VERSION=2.11.0"
    if "%MILO_TORCHVISION_VERSION%"=="" set "MILO_TORCHVISION_VERSION=0.26.0"
    if "%MILO_TORCHAUDIO_VERSION%"=="" set "MILO_TORCHAUDIO_VERSION=2.11.0"
    exit /b 0
)
if /I "%MILO_TORCH_CUDA%"=="cu121" (
    set "REQUIRED_CUDA_VERSION=12.1"
    set "REQUIRED_CUDA_WINGET_VERSION=12.1.0"
    set "REQUIRED_CUDA_INSTALLER_FILE=cuda_12.1.1_531.14_windows.exe"
    set "REQUIRED_CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_531.14_windows.exe"
    set "MILO_CUDA_VERSION=12.1"
    if "%MILO_TORCH_VERSION%"=="" set "MILO_TORCH_VERSION=2.3.1"
    if "%MILO_TORCHVISION_VERSION%"=="" set "MILO_TORCHVISION_VERSION=0.18.1"
    if "%MILO_TORCHAUDIO_VERSION%"=="" set "MILO_TORCHAUDIO_VERSION=2.3.1"
    exit /b 0
)
if /I "%MILO_TORCH_CUDA%"=="cu118" (
    set "REQUIRED_CUDA_VERSION=11.8"
    set "REQUIRED_CUDA_WINGET_VERSION=11.8.0"
    set "REQUIRED_CUDA_INSTALLER_FILE=cuda_11.8.0_522.06_windows.exe"
    set "REQUIRED_CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe"
    set "MILO_CUDA_VERSION=11.8"
    if "%MILO_TORCH_VERSION%"=="" set "MILO_TORCH_VERSION=2.3.1"
    if "%MILO_TORCHVISION_VERSION%"=="" set "MILO_TORCHVISION_VERSION=0.18.1"
    if "%MILO_TORCHAUDIO_VERSION%"=="" set "MILO_TORCHAUDIO_VERSION=2.3.1"
    exit /b 0
)

echo.
echo Unsupported MILO_TORCH_CUDA value: %MILO_TORCH_CUDA%
echo Use cu128, cu121, or cu118.
exit /b 1

:ensure_cpp_build_tools
where cl >nul 2>&1
if not errorlevel 1 exit /b 0

call :activate_msvc_tools
where cl >nul 2>&1
if not errorlevel 1 exit /b 0

where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo Microsoft C++ Build Tools were not found, and winget is not available for automatic install.
    echo Install Visual Studio Build Tools with Desktop development with C++, then run start.bat again.
    exit /b 1
)

echo Microsoft C++ Build Tools were not found. Installing with winget...
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
if errorlevel 1 exit /b 1

call :activate_msvc_tools
where cl >nul 2>&1
if not errorlevel 1 exit /b 0

echo.
echo Microsoft C++ Build Tools were installed, but cl.exe was not found in this shell.
echo Open "x64 Native Tools Command Prompt for VS 2022" and run start.bat again.
exit /b 1

:activate_msvc_tools
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%I\Common7\Tools\VsDevCmd.bat" (
        call "%%I\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
        exit /b 0
    )
)
exit /b 0

:failed
echo.
echo Startup failed.
echo Press any key to exit.
pause
exit /b 1

:finished
echo.
echo Server stopped.
echo Press any key to exit.
pause
exit /b 0
