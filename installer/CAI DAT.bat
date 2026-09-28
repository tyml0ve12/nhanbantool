@echo off
REM Bam dup file nay de cai dat "Nhan Ban Long Tieng".
REM File nay chi la lop vo goi PowerShell chay _files\install.ps1 voi quyen
REM thuc thi tam thoi (Bypass CHI cho tien trinh nay, KHONG doi chinh sach
REM chung cua may).

setlocal
set SCRIPT_DIR=%~dp0

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%_files\install.ps1"

echo.
pause
