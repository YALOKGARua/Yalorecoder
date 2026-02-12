@echo off
echo ========================================
echo   Building Audio Recorder
echo   by YALOKGAR
echo ========================================
echo.

echo Resolving paths...
for /f "delims=" %%i in ('python -c "import customtkinter, os; print(os.path.dirname(customtkinter.__file__))"') do set CTK_PATH=%%i
for /f "delims=" %%i in ('python -c "import sys, os; print(os.path.dirname(sys.executable))"') do set PY_DIR=%%i
echo   CTK: %CTK_PATH%
echo   Python: %PY_DIR%
echo.

if not exist "icon.ico" (
    echo WARNING: icon.ico not found, building without icon
    set ICON_ARGS=
    set ICON_DATA=
) else (
    set ICON_ARGS=--icon "icon.ico"
    set ICON_DATA=--add-data "icon.ico;."
)

echo [1/2] Building executable...
pyinstaller --noconfirm --onefile --windowed ^
    --name "AudioRecorder" ^
    %ICON_ARGS% ^
    --uac-admin ^
    %ICON_DATA% ^
    --add-binary "%PY_DIR%\vcruntime140.dll;." ^
    --add-binary "%PY_DIR%\vcruntime140_1.dll;." ^
    --add-data "%CTK_PATH%;customtkinter" ^
    --hidden-import "soundfile" ^
    --hidden-import "pystray._win32" ^
    --collect-all "_soundfile_data" ^
    recorder.py

if not exist "dist\AudioRecorder.exe" (
    echo ERROR: exe build failed
    pause
    exit /b 1
)
echo.

echo [2/2] Building installer...
set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"

if exist %ISCC% (
    %ISCC% "installer.iss"
) else (
    echo WARNING: Inno Setup not found, skipping installer
)

echo.
echo ========================================
echo   Build complete!
echo   EXE: dist\AudioRecorder.exe
if exist "dist\AudioRecorder_Setup_v*.exe" echo   Setup: dist\AudioRecorder_Setup_v*.exe
echo ========================================
pause
