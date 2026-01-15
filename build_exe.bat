@echo off
REM Nettoyage des anciens dossiers build et dist
rmdir /S /Q build
rmdir /S /Q dist

echo Nettoyage terminé.

REM Compilation avec PyInstaller
C:\Users\lavig\AppData\Local\Programs\Python\Python312\Scripts\pyinstaller.exe --onefile --add-data "yolov3-tiny.cfg;." --add-data "yolov3-tiny.weights;." --add-data "coco.names;." main.py

echo.
echo Si tout s'est bien passé, l'exécutable se trouve dans le dossier dist\main.exe
pause
