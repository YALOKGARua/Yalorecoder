@echo off
echo ========================================
echo   Building Audio Recorder .exe
echo   by YALOKGAR
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt
echo.

echo Building executable...
pyinstaller --noconfirm --onefile --windowed ^
    --name "AudioRecorder" ^
    --icon "icon.ico" ^
    --add-data "icon.ico;." ^
    --add-data "icon.png;." ^
    --add-data "%PYTHON%\Lib\site-packages\customtkinter;customtkinter" ^
    recorder.py

if not exist "dist\AudioRecorder.exe" (
    echo Retrying with explicit path...
    for /f "delims=" %%i in ('python -c "import customtkinter; import os; print(os.path.dirname(customtkinter.__file__))"') do set CTK_PATH=%%i
    pyinstaller --noconfirm --onefile --windowed ^
        --name "AudioRecorder" ^
        --icon "icon.ico" ^
        --add-data "icon.ico;." ^
        --add-data "icon.png;." ^
        --add-data "%CTK_PATH%;customtkinter" ^
        recorder.py
)

echo.
echo ========================================
echo   Build complete!
echo   Executable: dist\AudioRecorder.exe
echo ========================================
pause
