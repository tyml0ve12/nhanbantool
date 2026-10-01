# Cai dat Nhan Ban Long Tieng - khong dung exe tu build (PyInstaller), tranh bi
# Smart App Control / Windows Defender chan vi "unknown publisher, no reputation".
# (Theo HUONG_DAN_BUILD_VA_UPDATE.md, mau tu Tool Join Video.)
#
# Uu tien tu tren xuong:
#  1. May da co Python kem day du thu vien (PySide6, faster-whisper) -> dung luon.
#  2. May da co Python (3.10+, 64-bit, co pip) nhung thieu thu vien -> chi
#     pip install vao Python do (--user, khong can admin).
#  3. May chua co Python nao dung duoc -> tai Python chinh chu tu python.org
#     cai rieng cho tool, roi pip install vao do.
# Shortcut chay thang qua pythonw.exe (da ky so) -> Windows khong chan.
#
# Idempotent: chay lai = bo qua phan da co, copy code moi, tao lai shortcut.
# models/ va bin/ (~2.5GB) chi copy file thay doi (robocopy), khong copy lai het.

$ErrorActionPreference = "Stop"

$PythonVersionCandidates = @("3.12.6", "3.13.15", "3.11.9")
$AppName = "Nhan Ban Long Tieng"

$AppRoot = Join-Path $env:LOCALAPPDATA "NhanBanLongTieng"
$PyDir = Join-Path $AppRoot "python"
$AppDir = Join-Path $AppRoot "app"
$PythonwExe = $null

# Code + models + bin nam CUNG thu muc voi script nay (thu muc "_files").
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $SourceDir "requirements.txt"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# ---------------- Buoc 1: Tim / chuan bi Python + thu vien ----------------
#
# Su co thuc te (2026-09-19): Windows Installer coi "Python 3.12" la 1 SAN PHAM
# DUY NHAT bat ke TargetDir - may da co 3.12 o cho khac ma cai them 1 ban 3.12
# rieng thi installer bao thanh cong nhung KHONG COPY FILE NAO. Vi vay uu tien
# Python co san, neu phai tai rieng thi chon ban co major.minor chua co tren may.

function Test-PythonRuns($pythonExe, $code) {
    if (-not (Test-Path $pythonExe)) { return $false }
    try {
        & $pythonExe -c $code *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

# faster-whisper < 1.1.1 goi av.open(metadata_errors=...) - PyAV 14+ da bo tham so nay
# -> loi "unexpected keyword argument 'metadata_errors'". Ban cu coi nhu thieu thu vien
# de buoc 1b pip install -r requirements.txt nang cap len.
function Test-LibsWork($pythonExe) {
    return Test-PythonRuns $pythonExe "import re, PySide6.QtWidgets, faster_whisper, onnxruntime, tokenizers; assert tuple(int(x) for x in re.findall(r'\d+', faster_whisper.__version__)[:3]) >= (1, 1, 1)"
}

# PySide6 / ctranslate2 chi co ban cai san cho Python 64-bit, 3.10 tro len.
function Test-PythonSuitable($pythonExe) {
    return Test-PythonRuns $pythonExe "import sys, struct; assert sys.version_info >= (3, 10) and struct.calcsize('P') == 8; import pip"
}

function Get-RegisteredPythons {
    $result = New-Object System.Collections.Generic.List[object]
    $roots = @(
        "HKCU:\Software\Python\PythonCore",
        "HKLM:\Software\Python\PythonCore",
        "HKLM:\Software\WOW6432Node\Python\PythonCore"
    )
    foreach ($root in $roots) {
        foreach ($key in (Get-ChildItem $root -ErrorAction SilentlyContinue)) {
            $installDir = $null
            $ipKey = Join-Path $key.PSPath "InstallPath"
            if (Test-Path $ipKey) {
                $installDir = (Get-ItemProperty $ipKey -ErrorAction SilentlyContinue).'(default)'
            }
            $result.Add([pscustomobject]@{ Tag = $key.PSChildName; InstallDir = $installDir })
        }
    }
    return $result
}

# Danh sach thu muc Python (co ca python.exe + pythonw.exe), Python rieng cua tool dung dau.
function Find-PythonDirs($registeredPythons) {
    $candidateDirs = New-Object System.Collections.Generic.List[string]
    $candidateDirs.Add($PyDir)
    foreach ($py in $registeredPythons) {
        if ($py.InstallDir) { $candidateDirs.Add($py.InstallDir.TrimEnd('\')) }
    }
    foreach ($cmdName in @("pythonw.exe", "python.exe")) {
        foreach ($cmd in (Get-Command $cmdName -All -ErrorAction SilentlyContinue)) {
            $candidateDirs.Add((Split-Path $cmd.Source -Parent))
        }
    }
    $dirPatterns = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*"),
        "C:\Python3*",
        (Join-Path ${env:ProgramFiles} "Python3*")
    )
    foreach ($pattern in $dirPatterns) {
        # Get-Item (khong phai Get-ChildItem) - Get-ChildItem liet ke thu muc CON, bo sot chinh thu muc Python
        foreach ($d in (Get-Item -Path $pattern -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer })) {
            $candidateDirs.Add($d.FullName)
        }
    }

    $result = New-Object System.Collections.Generic.List[string]
    $seen = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::OrdinalIgnoreCase)
    foreach ($dir in $candidateDirs) {
        if (-not $dir -or -not $seen.Add($dir)) { continue }
        # Bo ban gia cua Microsoft Store (WindowsApps) - chay im lang, khong lam gi
        if ($dir -like "*\WindowsApps*") { continue }
        if ((Test-Path (Join-Path $dir "pythonw.exe")) -and (Test-Path (Join-Path $dir "python.exe"))) {
            $result.Add($dir)
        }
    }
    return $result
}

function Install-Libs($pythonExe, [switch]$UserSite) {
    $pipArgs = @("-m", "pip", "install", "--disable-pip-version-check", "-r", $Requirements)
    if ($UserSite) { $pipArgs += "--user" }
    & $pythonExe @pipArgs
    return ($LASTEXITCODE -eq 0) -and (Test-LibsWork $pythonExe)
}

$registeredPythons = Get-RegisteredPythons
$pythonDirs = Find-PythonDirs $registeredPythons

# 1a. Python da co san du thu vien
foreach ($dir in $pythonDirs) {
    if (Test-LibsWork (Join-Path $dir "python.exe")) {
        $PythonwExe = Join-Path $dir "pythonw.exe"
        Write-Step "May da co san Python kem day du thu vien tai '$dir' - dung luon."
        break
    }
}

# 1b. Python co san nhung thieu thu vien -> cai them vao do
if (-not $PythonwExe) {
    foreach ($dir in $pythonDirs) {
        $py = Join-Path $dir "python.exe"
        if (-not (Test-PythonSuitable $py)) { continue }
        Write-Step "May da co Python tai '$dir' - cai them thu vien (khoang 300MB, co the mat vai phut)..."
        $isOwnPython = $dir -eq $PyDir
        if (Install-Libs $py -UserSite:(-not $isOwnPython)) {
            $PythonwExe = Join-Path $dir "pythonw.exe"
            Write-Host "Cai thu vien thanh cong." -ForegroundColor Green
            break
        }
        Write-Host "Khong cai duoc thu vien vao Python nay, thu cach khac..." -ForegroundColor Yellow
    }
}

# 1c. Chua co Python nao dung duoc -> tai Python rieng cho tool
if (-not $PythonwExe) {
    $registeredMinors = @($registeredPythons | ForEach-Object { ($_.Tag -split '-')[0] })
    $PythonVersion = $null
    foreach ($v in $PythonVersionCandidates) {
        $minor = ($v -split '\.')[0..1] -join '.'
        if ($registeredMinors -notcontains $minor) { $PythonVersion = $v; break }
    }
    if (-not $PythonVersion) {
        Write-Host "LOI: May da co Python $($registeredMinors -join ', ') nhung khong ban nao cai duoc thu vien," -ForegroundColor Red
        Write-Host "va khong con phien ban du phong nao de cai rieng ma khong dung do." -ForegroundColor Red
        Write-Host "Kiem tra ket noi internet roi chay lai file nay." -ForegroundColor Yellow
        exit 1
    }

    # Don thu muc python rieng bi rong/hong tu lan cai loi truoc.
    if ((Test-Path $PyDir) -and -not (Test-Path (Join-Path $PyDir "pythonw.exe"))) {
        Remove-Item $PyDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    $PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    Write-Step "Chua co Python dung duoc - dang tai Python $PythonVersion chinh chu tu python.org (khoang 25-30MB)..."
    New-Item -ItemType Directory -Force -Path $AppRoot | Out-Null
    $InstallerPath = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    try {
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $InstallerPath -UseBasicParsing
    } catch {
        Write-Host "LOI: Khong tai duoc bo cai Python. Kiem tra ket noi internet roi thu lai." -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }

    $LogPath = Join-Path $env:TEMP "nhanban_python_install.log"
    Write-Step "Dang cai Python rieng cho tool (khong can quyen admin, khong anh huong Python khac tren may)..."
    $installArgs = @(
        "/quiet",
        "/log", "`"$LogPath`"",
        "InstallAllUsers=0",
        "TargetDir=`"$PyDir`"",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "Include_tcltk=0",
        "Include_pip=1"
    )
    $proc = Start-Process -FilePath $InstallerPath -ArgumentList $installArgs -Wait -PassThru
    Remove-Item $InstallerPath -ErrorAction SilentlyContinue

    $ownPython = Join-Path $PyDir "python.exe"
    # Kiem tra THUC SU chay duoc, khong tin exit code (xem su co 2026-09-19).
    if ($proc.ExitCode -ne 0 -or -not (Test-PythonSuitable $ownPython)) {
        Write-Host "LOI: Cai Python that bai (exit code $($proc.ExitCode))." -ForegroundColor Red
        Write-Host "Chi tiet log cai dat: $LogPath" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Cai Python $PythonVersion thanh cong tai '$PyDir'." -ForegroundColor Green

    Write-Step "Dang cai thu vien (khoang 300MB, co the mat vai phut)..."
    if (-not (Install-Libs $ownPython)) {
        Write-Host "LOI: Cai thu vien that bai. Kiem tra ket noi internet roi chay lai file nay." -ForegroundColor Red
        exit 1
    }
    Write-Host "Cai thu vien thanh cong." -ForegroundColor Green
    $PythonwExe = Join-Path $PyDir "pythonw.exe"
}

# ---------------- Buoc 2: Copy code + models + bin vao noi cai dat ----------------

Write-Step "Dang sao chep tool vao '$AppDir' (lan dau mat vai phut vi model ~2GB)..."
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null

foreach ($item in @("main.py", "cli.py", "requirements.txt", "core", "ui")) {
    $src = Join-Path $SourceDir $item
    if (-not (Test-Path $src)) {
        Write-Host "LOI: Khong tim thay '$src'. Kiem tra lai cau truc thu muc phat hanh." -ForegroundColor Red
        exit 1
    }
    $dst = Join-Path $AppDir $item
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    Copy-Item -Path $src -Destination $AppDir -Recurse -Force
}

# File nang: robocopy chi copy file moi/thay doi (cai lai khong phai copy lai 2.5GB).
foreach ($item in @("models", "bin")) {
    $src = Join-Path $SourceDir $item
    if (-not (Test-Path $src)) {
        Write-Host "LOI: Khong tim thay '$src' - ban phat hanh thieu file." -ForegroundColor Red
        exit 1
    }
    robocopy $src (Join-Path $AppDir $item) /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Write-Host "LOI: Sao chep '$item' that bai (robocopy $LASTEXITCODE). O dia con du cho trong khong?" -ForegroundColor Red
        exit 1
    }
}

# Danh dau day la ban da cai - app chi bat tu cap nhat khi co file nay.
Set-Content -Path (Join-Path $AppDir ".installed") -Value "installed by CAI DAT.bat" -Encoding ascii
Write-Host "Da sao chep xong." -ForegroundColor Green

# ---------------- Buoc 3: Tao shortcut (Desktop + Start Menu) ----------------

Write-Step "Dang tao shortcut..."

function New-AppShortcut($shortcutPath) {
    $wshell = New-Object -ComObject WScript.Shell
    $shortcut = $wshell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $PythonwExe
    $shortcut.Arguments = "`"$AppDir\main.py`""
    $shortcut.WorkingDirectory = $AppDir
    $shortcut.Description = "$AppName - BrightStar - Hoang Duc"
    $shortcut.Save()
}

New-AppShortcut (Join-Path ([Environment]::GetFolderPath("Desktop")) "$AppName.lnk")
New-AppShortcut (Join-Path (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs") "$AppName.lnk")
Write-Host "Da tao shortcut tai Desktop va Start Menu." -ForegroundColor Green

Write-Step "HOAN TAT! Mo '$AppName' tu Desktop hoac Start Menu de dung."
