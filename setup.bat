REM Se placer dans le dossier du script
cd /d %~dp0
@echo off
REM setup.bat - Simple setup script for people-counter project

REM Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip



REM Install all requirements (force reinstall to ensure all are present)
python -m pip install --upgrade --force-reinstall -r requirements.txt

echo Setup complete. To activate the environment later, run:
echo     venv\Scripts\activate
