@echo off
REM Quick Start Script for EmotiSense AI (Windows)

cls
echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║         🎤 EMOTISENSE AI - QUICK START                ║
echo ║     Indian Speech Emotion Recognition System           ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM Check Python
echo [1] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
python --version
echo ✅ Python found
echo.

REM Check files
echo [2] Checking project files...
if not exist "app.py" (
    echo ❌ Missing: app.py
    pause
    exit /b 1
)
if not exist "index.html" (
    echo ❌ Missing: index.html
    pause
    exit /b 1
)
if not exist "best_model_weights.h5" (
    echo ❌ Missing: best_model_weights.h5
    pause
    exit /b 1
)
if not exist "scaler.pickle" (
    echo ❌ Missing: scaler.pickle
    pause
    exit /b 1
)
echo ✅ All files found
echo.

REM Install dependencies
echo [3] Installing dependencies (this may take 2-3 minutes)...
pip install -r requirements.txt --break-system-packages -q
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed
echo.

REM Verify models
echo [4] Verifying model files...
python -c "import h5py; import pickle; import json; h5py.File('best_model_weights.h5', 'r').close(); pickle.load(open('scaler.pickle', 'rb')); print('✅ All models verified')"
if errorlevel 1 (
    echo ❌ Model verification failed
    pause
    exit /b 1
)
echo.

REM Start server
echo [5] Starting EmotiSense server...
echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║  🚀 Server running at: http://localhost:5000          ║
echo ║                                                        ║
echo ║  1. Open browser to http://localhost:5000            ║
echo ║  2. Click 'Start Recording'                          ║
echo ║  3. Speak for 2-3 seconds                            ║
echo ║  4. View emotion detection results                   ║
echo ║                                                        ║
echo ║  Press Ctrl+C to stop the server                    ║
echo ╚════════════════════════════════════════════════════════╝
echo.

python app.py
pause
