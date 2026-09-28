@echo off
cd /d "%~dp0"

rem Ban dung cho may phat trien. May co nhieu ban Python cai song song, va
rem Explorer co the giu PATH cu -> "pythonw" tro vao ban gia cua Microsoft Store
rem (thu muc WindowsApps) va khong lam gi ca. Vi vay do tung ban, chon ban
rem import duoc PySide6 + faster_whisper.
set "PYW="

for %%P in ("C:\Python314\pythonw.exe" "%LocalAppData%\Python\bin\pythonw.exe") do (
    if not defined PYW if exist "%%~P" (
        "%%~dpPpython.exe" -c "import PySide6, faster_whisper" >nul 2>&1 && set "PYW=%%~P"
    )
)

if not defined PYW (
    for /f "delims=" %%P in ('where pythonw 2^>nul ^| findstr /v /i "WindowsApps"') do (
        if not defined PYW (
            "%%~dpPpython.exe" -c "import PySide6, faster_whisper" >nul 2>&1 && set "PYW=%%P"
        )
    )
)

if not defined PYW (
    echo Khong tim thay ban Python nao da cai PySide6 va faster-whisper.
    echo Hay chay: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "" "%PYW%" main.py
