# PowerShell — Download datasets using aria2c (and gdown for Google Drive)
# Run from repo root:  .\scripts\download_datasets.ps1

$dataDir = "data\raw"

# ── Helper ──────────────────────────────────────────────────
function Ensure-Dir($path) { if (!(Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null } }

# ── 1. IIIT 5K-Word  (~200 MB, direct HTTP) ────────────────
Write-Output "`n>>> [1/4] Downloading IIIT 5K-Word..."
Ensure-Dir "$dataDir\iiit5k"
aria2c -x 8 -s 8 -d "$dataDir\iiit5k" -o "IIIT5K-Word_V3.0.tar.gz" `
  "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz"
if (Test-Path "$dataDir\iiit5k\IIIT5K-Word_V3.0.tar.gz") {
    Write-Output "Extracting IIIT 5K-Word..."
    tar -xzf "$dataDir\iiit5k\IIIT5K-Word_V3.0.tar.gz" -C "$dataDir\iiit5k"
}

# ── 2. MJSynth / Synth90k  (~10 GB, direct HTTP) ──────────
Write-Output "`n>>> [2/4] Downloading MJSynth (Synth90k) — ~10 GB, be patient..."
Ensure-Dir "$dataDir\mjsynth"
aria2c -x 8 -s 8 -d "$dataDir\mjsynth" -o "mjsynth.tar.gz" `
  "https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz"
if (Test-Path "$dataDir\mjsynth\mjsynth.tar.gz") {
    Write-Output "Extracting MJSynth..."
    tar -xzf "$dataDir\mjsynth\mjsynth.tar.gz" -C "$dataDir\mjsynth"
}

# ── 3. Total-Text  (Google Drive — need gdown) ─────────────
Write-Output "`n>>> [3/4] Total-Text — cloning repo for annotation files..."
Ensure-Dir "$dataDir\totaltext"
git clone --depth 1 "https://github.com/cs-chan/Total-Text-Dataset.git" "$dataDir\totaltext\repo" 2>$null
Write-Output @"
Total-Text images are on Google Drive. Open the repo README for links:
  $dataDir\totaltext\repo\README.md
Then download images into:
  $dataDir\totaltext\Images\Train\
  $dataDir\totaltext\Images\Test\
Or use gdown if you have the file IDs:
  gdown <GOOGLE_DRIVE_FILE_ID> -O $dataDir\totaltext\images.zip
"@

# ── 4. CTW1500  (Google Drive — need gdown) ─────────────────
Write-Output "`n>>> [4/4] CTW1500 — cloning repo for annotation files..."
Ensure-Dir "$dataDir\ctw1500"
git clone --depth 1 "https://github.com/Yuliang-Liu/Curve-Text-Detector.git" "$dataDir\ctw1500\repo" 2>$null
Write-Output @"
CTW1500 images are on Google Drive. Open the repo README for links:
  $dataDir\ctw1500\repo\README.md
Then download images into:
  $dataDir\ctw1500\train\text_image\
  $dataDir\ctw1500\test\text_image\
"@

# ── Manual-download reminders ────────────────────────────────
Write-Output @"

========================================
  DATASETS THAT NEED MANUAL DOWNLOAD
========================================
5. ICDAR 2015  → Register at https://rrc.cvc.uab.es/ (Challenge 4)
     Place in: $dataDir\icdar2015\
6. IIIT-H Indic Scene Text → Request from https://cvit.iiit.ac.in/
     Place in: $dataDir\indic_scene\
7. IDD (Indian Driving) → Register at https://idd.insaan.iiit.ac.in/
     Place in: $dataDir\idd\
8. SynthText (OPTIONAL, 41 GB) → aria2c -x8 -s8 -d $dataDir\synthtext -o SynthText.zip https://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip
========================================
"@

Write-Output "Done. Check $dataDir\ for downloaded files."
