echo Nettoyage terminé.
echo.
echo Si tout s'est bien passé, l'exécutable se trouve dans le dossier dist\main.exe

@echo off
REM Nettoyage des anciens dossiers build et dist
rmdir /S /Q build
rmdir /S /Q dist

echo Nettoyage terminé.

REM Activation de l'environnement virtuel
call venv\Scripts\activate

REM Installation de PyInstaller dans le venv si nécessaire
pip show pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
	echo Installation de PyInstaller...
	pip install pyinstaller
)

REM Compilation avec PyInstaller du venv
pyinstaller --onefile --hidden-import=PyQt5.sip --hidden-import=PyQt5.QtWidgets --hidden-import=PyQt5.QtGui --hidden-import=PyQt5.QtCore --hidden-import=cv2 main.py

echo.
echo Si tout s'est bien passé, l'exécutable et les fichiers nécessaires se trouvent dans le dossier dist\
pause
