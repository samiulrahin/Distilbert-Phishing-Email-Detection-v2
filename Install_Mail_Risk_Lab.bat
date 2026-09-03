@echo off
cd /d "%~dp0"
title Mail Risk Lab - First-time installation
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_portable.ps1"
if errorlevel 1 (
  echo.
  echo Installation did not complete. Read PORTABLE_PACKAGE_README.md.
) else (
  echo.
  echo Installation complete. Double-click Start_Mail_Risk_Lab.bat.
)
pause
