@echo off
echo ================================================================
echo LOOPYCOMFY COMFYUI DEPENDENCY INSTALLER
echo ================================================================
echo.
echo This script will install missing dependencies in your ComfyUI environment.
echo Make sure ComfyUI is closed before running this script.
echo.
pause

REM Change to ComfyUI directory - adjust path if different
echo Navigating to ComfyUI directory...
cd /d "I:\ComfyUI_windows_portable"

REM Check if python_embeded exists
if not exist "python_embeded\python.exe" (
    echo ERROR: python_embeded\python.exe not found!
    echo Please adjust the path in this script to match your ComfyUI installation.
    echo Current path: I:\ComfyUI_windows_portable
    pause
    exit /b 1
)

echo Found ComfyUI Python: python_embeded\python.exe
echo.

REM Upgrade pip first
echo Upgrading pip...
.\python_embeded\python.exe -m pip install --upgrade pip
echo.

REM Install missing dependencies one by one
echo Installing core dependencies...
.\python_embeded\python.exe -m pip install "numpy>=1.24.0,<2.0.0"
if %errorlevel% neq 0 echo WARNING: numpy installation had issues but continuing...

.\python_embeded\python.exe -m pip install "opencv-python>=4.8.0,<5.0.0"
if %errorlevel% neq 0 echo WARNING: opencv-python installation had issues but continuing...

.\python_embeded\python.exe -m pip install "psutil>=5.9.6,<6.0.0"
if %errorlevel% neq 0 echo WARNING: psutil installation had issues but continuing...

.\python_embeded\python.exe -m pip install "scipy>=1.10.0,<2.0.0"
if %errorlevel% neq 0 echo WARNING: scipy installation had issues but continuing...

.\python_embeded\python.exe -m pip install "Pillow>=10.0.0,<11.0.0"
if %errorlevel% neq 0 echo WARNING: Pillow installation had issues but continuing...

.\python_embeded\python.exe -m pip install "imageio>=2.30.0,<3.0.0"
if %errorlevel% neq 0 echo WARNING: imageio installation had issues but continuing...

.\python_embeded\python.exe -m pip install "ffmpeg-python>=0.2.0,<1.0.0"
if %errorlevel% neq 0 echo WARNING: ffmpeg-python installation had issues but continuing...

.\python_embeded\python.exe -m pip install "scikit-learn>=1.3.0,<2.0.0"
if %errorlevel% neq 0 echo WARNING: scikit-learn installation had issues but continuing...

echo.
echo ================================================================
echo TESTING INSTALLATION
echo ================================================================

REM Test the installation
echo Testing dependencies...
.\python_embeded\python.exe -c "
import sys
import os
import importlib

# Add loopy-comfy to path
sys.path.insert(0, os.path.join(os.getcwd(), 'ComfyUI', 'custom_nodes', 'loopy-comfy'))

print('Testing dependencies:')
deps = ['numpy', 'cv2', 'psutil', 'scipy', 'PIL', 'imageio', 'ffmpeg', 'sklearn']
all_good = True

for dep in deps:
    try:
        importlib.import_module(dep)
        print(f'  ✓ {dep} - OK')
    except ImportError as e:
        print(f'  ✗ {dep} - MISSING')
        all_good = False

print()
if all_good:
    print('SUCCESS: All dependencies are now installed!')
    print('You can now restart ComfyUI and the nodes should load.')
else:
    print('Some dependencies are still missing. Check the errors above.')
"

echo.
echo ================================================================
echo INSTALLATION COMPLETE
echo ================================================================
echo.
echo Next steps:
echo 1. Start ComfyUI
echo 2. Check if LoopyComfy nodes now appear in the node browser
echo 3. If you still see errors, check the ComfyUI console for specific messages
echo.
pause