@echo off
cd /d "%~dp0"
title Mail Risk Lab - Package verification
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\verify_portable.ps1"
if errorlevel 1 echo Verification failed. Read PORTABLE_PACKAGE_README.md.
pause
