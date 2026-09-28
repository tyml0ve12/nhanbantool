# Script CHO NGUOI PHAT TRIEN - KHONG gui cho nguoi dung cuoi.
#
# Moi lan ra ban moi:
#   1. Sua APP_VERSION trong core\version.py (vd "1.1")
#   2. Chay:  powershell -ExecutionPolicy Bypass -File make_release.ps1 -Notes "Sua loi X, them Y"
#      Mac dinh cap nhat la BAT BUOC. Them -Optional neu cho phep bam "De sau".
#      Them -UpdateOnly neu chi can goi cap nhat (khong dong goi lai bo cai 2.5GB).
#
# Ket qua trong dist\:
#   NhanBanLongTieng_v<ver>\ + .zip  - bo cai DAY DU (code + model + CUDA + ffmpeg)
#                                      -> dua len GOOGLE DRIVE cho nguoi dung MOI
#   release_v<ver>\                   - dang len GITHUB RELEASE "v<ver>" (chi code, vai tram KB):
#       app_update.zip, latest.json   -> app cua nguoi dung da cai tu cap nhat

param(
    [string]$Notes = "",
    [switch]$Optional,
    [switch]$UpdateOnly
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

$versionFile = Get-Content (Join-Path $Root "core\version.py") -Raw
$Version = [regex]::Match($versionFile, 'APP_VERSION\s*=\s*"([^"]+)"').Groups[1].Value
$Repo = [regex]::Match($versionFile, 'GITHUB_REPO\s*=\s*"([^"]*)"').Groups[1].Value
if (-not $Version) { throw "Khong doc duoc APP_VERSION trong core\version.py" }

$CodeItems = @("main.py", "cli.py", "requirements.txt", "core", "ui")
# Chi dong goi model dang dung (khong kem large-v3, tiny... tai ve luc thu nghiem)
$ModelDirs = @("models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", "multilingual-minilm")
$CudaDlls = @("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")

function Copy-CodeItems($dest) {
    foreach ($item in $CodeItems) {
        Copy-Item -Path (Join-Path $Root $item) -Destination $dest -Recurse -Force
    }
    Get-ChildItem -Path $dest -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
}

function Find-Tool($name) {
    $bundled = Join-Path $Root "bin\$name"
    if (Test-Path $bundled) { return $bundled }
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Khong tim thay $name (dat vao bin\ hoac cai FFmpeg vao PATH)."
}

# ---------- 1. Bo cai day du -> Google Drive ----------
$ReleaseName = "NhanBanLongTieng_v$Version"
$DistApp = Join-Path $Root "dist\$ReleaseName"
$FilesDir = Join-Path $DistApp "_files"

if (-not $UpdateOnly) {
    if (Test-Path $DistApp) { Remove-Item $DistApp -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $FilesDir | Out-Null
    Copy-CodeItems $FilesDir
    Copy-Item -Path (Join-Path $Root "installer\install.ps1") -Destination $FilesDir -Force
    Copy-Item -Path (Join-Path $Root "installer\CAI DAT.bat") -Destination $DistApp -Force
    Set-Content -Path (Join-Path $DistApp "VERSION.txt") -Value "Nhan Ban Long Tieng - phien ban v$Version" -Encoding utf8

    Write-Host "Dang chep model..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Join-Path $FilesDir "models") | Out-Null
    foreach ($m in $ModelDirs) {
        $src = Join-Path $Root "models\$m"
        if (-not (Test-Path $src)) { throw "Thieu model '$src' - chay app 1 lan de tai model truoc." }
        robocopy $src (Join-Path $FilesDir "models\$m") /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Chep model $m that bai" }
    }

    Write-Host "Dang chep ffmpeg + CUDA..." -ForegroundColor Cyan
    $BinDir = Join-Path $FilesDir "bin"
    New-Item -ItemType Directory -Force -Path (Join-Path $BinDir "cuda") | Out-Null
    Copy-Item (Find-Tool "ffmpeg.exe") $BinDir
    Copy-Item (Find-Tool "ffprobe.exe") $BinDir
    foreach ($dll in $CudaDlls) {
        $src = Join-Path $Root "bin\cuda\$dll"
        if (-not (Test-Path $src)) { throw "Thieu '$src' (3 DLL CUDA whisper can)." }
        Copy-Item $src (Join-Path $BinDir "cuda")
    }

    $InstallerZip = Join-Path $Root "dist\$ReleaseName.zip"
    if (Test-Path $InstallerZip) { Remove-Item $InstallerZip -Force }
    Write-Host "Dang nen zip (~2.5GB, mat vai phut)..." -ForegroundColor Cyan
    # tar.exe co san tu Windows 10 - Compress-Archive cua PowerShell 5.1 loi voi file > 2GB
    Push-Location (Join-Path $Root "dist")
    tar -a -c -f "$ReleaseName.zip" "$ReleaseName"
    Pop-Location
    if ($LASTEXITCODE -ne 0) { throw "Nen zip that bai" }
    $sizeGB = [math]::Round((Get-Item $InstallerZip).Length / 1GB, 2)
    Write-Host "Bo cai day du ($sizeGB GB) -> dua len Google Drive: $InstallerZip" -ForegroundColor Green
}

# ---------- 2. Goi cap nhat -> GitHub Release ----------
if (-not $Repo) {
    Write-Host "GITHUB_REPO trong core\version.py dang de trong - bo qua tao goi cap nhat." -ForegroundColor Yellow
    exit 0
}

$ReleaseDir = Join-Path $Root "dist\release_v$Version"
$StageDir = Join-Path $env:TEMP "nhanban_update_stage"
foreach ($d in @($ReleaseDir, $StageDir)) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }
New-Item -ItemType Directory -Force -Path $ReleaseDir, $StageDir | Out-Null

Copy-CodeItems $StageDir
$UpdateZip = Join-Path $ReleaseDir "app_update.zip"
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $UpdateZip
Remove-Item $StageDir -Recurse -Force

$Tag = "v$Version"
$manifest = [ordered]@{
    version   = $Version
    notes     = $Notes
    url       = "https://github.com/$Repo/releases/download/$Tag/app_update.zip"
    sha256    = (Get-FileHash $UpdateZip -Algorithm SHA256).Hash.ToLower()
    mandatory = -not $Optional
}
$manifest | ConvertTo-Json | Set-Content -Path (Join-Path $ReleaseDir "latest.json") -Encoding utf8

Write-Host "Goi cap nhat cho GitHub Release '$Tag': $ReleaseDir" -ForegroundColor Green
Write-Host ""
Write-Host "Dang len GitHub: vao https://github.com/$Repo/releases/new" -ForegroundColor Cyan
Write-Host "  - Tag: $Tag   (bam 'Create new tag')" -ForegroundColor Cyan
Write-Host "  - Keo tha 2 file app_update.zip + latest.json vao o 'Attach binaries'" -ForegroundColor Cyan
Write-Host "  - Giu tick 'Set as the latest release', bam 'Publish release'" -ForegroundColor Cyan
