@echo off
setlocal enabledelayedexpansion
title W@Home Hive - Installer
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  W@HOME HIVE — Windows Installer                       ║
echo  ║  Akataleptos Distributed Spectral Search                ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ═══════════════════════════════════════════════
:: Check Python
:: ═══════════════════════════════════════════════
set PYTHON=
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo   [OK] Python !PYVER!
    set PYTHON=python
) else (
    where python3 >nul 2>&1
    if %errorlevel%==0 (
        for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set PYVER=%%v
        echo   [OK] Python !PYVER!
        set PYTHON=python3
    ) else (
        echo   [!!] Python not found.
        echo.
        echo   Download Python from: https://www.python.org/downloads/
        echo   IMPORTANT: Check "Add Python to PATH" during install.
        echo.
        echo   After installing Python, run this script again.
        pause
        exit /b 1
    )
)

:: ═══════════════════════════════════════════════
:: GPU Detection
:: ═══════════════════════════════════════════════
set HAS_CUDA=0
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo   [OK] GPU: %%g
        set HAS_CUDA=1
    )
) else (
    echo   [--] No NVIDIA GPU detected (CPU mode)
)
echo.

:: ═══════════════════════════════════════════════
:: Install Directory
:: ═══════════════════════════════════════════════
set INSTALL_DIR=%USERPROFILE%\.whome
echo   Install directory: %INSTALL_DIR%
echo.

if exist "%INSTALL_DIR%" (
    echo   Existing installation found. Updating...
) else (
    mkdir "%INSTALL_DIR%"
)

:: ═══════════════════════════════════════════════
:: Download Files
:: ═══════════════════════════════════════════════
echo   Downloading W@Home...
echo.

set SERVER=https://wathome.akataleptos.com
for %%f in (whome_gui.py client.py w_operator.py w_cuda.py screensaver.py) do (
    echo   Fetching %%f...
    curl -sSf "%SERVER%/static/%%f" -o "%INSTALL_DIR%\%%f" 2>nul
    if !errorlevel! neq 0 (
        if exist "%~dp0%%f" (
            copy /y "%~dp0%%f" "%INSTALL_DIR%\%%f" >nul
            echo   [OK] %%f (local copy)
        ) else (
            echo   [--] %%f skipped
        )
    ) else (
        echo   [OK] %%f
    )
)

:: ═══════════════════════════════════════════════
:: Virtual Environment
:: ═══════════════════════════════════════════════
echo.
echo   Setting up Python environment...

if not exist "%INSTALL_DIR%\venv" (
    %PYTHON% -m venv "%INSTALL_DIR%\venv"
)

call "%INSTALL_DIR%\venv\Scripts\activate.bat"
pip install --quiet --upgrade pip
pip install --quiet numpy scipy requests pystray Pillow

if %HAS_CUDA%==1 (
    echo   Installing CUDA support...
    pip install --quiet cupy-cuda12x 2>nul || pip install --quiet cupy-cuda11x 2>nul || echo   CuPy unavailable - using CPU
)

:: Optional screensaver deps (non-fatal if they fail)
pip install --quiet pygame moderngl pyrr 2>nul
if %errorlevel%==0 (
    echo   [OK] Screensaver support installed
) else (
    echo   [--] Screensaver deps unavailable (optional)
)

echo   [OK] Dependencies installed
echo.

:: ═══════════════════════════════════════════════
:: Create Launchers
:: ═══════════════════════════════════════════════

:: GUI launcher (default)
(
echo @echo off
echo cd /d "%INSTALL_DIR%"
echo call venv\Scripts\activate.bat
echo pythonw whome_gui.py %%*
) > "%INSTALL_DIR%\whome.bat"

:: CLI launcher (fallback)
(
echo @echo off
echo cd /d "%INSTALL_DIR%"
echo call venv\Scripts\activate.bat
echo python client.py %%*
) > "%INSTALL_DIR%\whome-cli.bat"

:: Desktop shortcut
set SHORTCUT=%USERPROFILE%\Desktop\W@Home.bat
copy /y "%INSTALL_DIR%\whome.bat" "%SHORTCUT%" >nul 2>nul
if %errorlevel%==0 echo   [OK] Desktop shortcut created

:: Start Menu shortcut
set STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs
copy /y "%INSTALL_DIR%\whome.bat" "%STARTMENU%\W@Home Hive.bat" >nul 2>nul
if %errorlevel%==0 echo   [OK] Start Menu shortcut created

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Installation Complete!                                  ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo   Launch GUI:          "%INSTALL_DIR%\whome.bat"
echo   CLI mode:            "%INSTALL_DIR%\whome-cli.bat"
echo   Or double-click:     W@Home on Desktop
echo   Uninstall:           rmdir /s "%INSTALL_DIR%"
echo.

set /p START="  Launch W@Home now? [Y/n]: "
if /i "!START!"=="n" (
    echo   Run whome.bat whenever you're ready.
) else (
    start "" "%INSTALL_DIR%\whome.bat"
)

pause
