@echo off
echo ────────────────────────────────────────────
echo  Resume Screening System - Windows Setup
echo ────────────────────────────────────────────

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (echo ERROR: Python not found. Install from https://python.org && pause && exit /b 1)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Installing PyTorch (CPU)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

echo [5/5] Installing dependencies...
pip install -r requirements.txt --quiet

echo Downloading spaCy model...
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --quiet

echo.
echo  Setup complete! Run: streamlit run layer7_app.py
pause