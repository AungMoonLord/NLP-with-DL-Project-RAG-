# ==============================================================================
# Pipeline Execution Script for Windows (PowerShell)
# ==============================================================================

$ErrorActionPreference = "Stop"

# บังคับใช้ Encoding UTF-8
$env:PYTHONUTF8 = 1
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Starting Full Legal-RAG Pipeline (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 0: ตรวจสอบ Tesseract OCR
# ------------------------------------------------------------------------------
Write-Host "`n>>> [0/5] Checking Tesseract OCR..." -ForegroundColor Yellow
$tessPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tessPath) {
    Write-Host "Tesseract OCR found at $tessPath" -ForegroundColor Green
} else {
    Write-Warning "Tesseract OCR not found at default path. Make sure it is installed if OCR fallback is needed."
}

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 1: Virtual Environment & Dependencies
# ------------------------------------------------------------------------------
Write-Host "`n>>> [1/5] Setting up Python Environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    py -m venv .venv
}

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements..."
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 2: Data Extraction (PDF -> Clean Text)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [2/5] Extracting & Cleaning Text from Legal PDFs..." -ForegroundColor Yellow
py src/extract_corpus.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 3: Chunking & Indexing (FAISS + BM25)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [3/5] Building Vector & BM25 Indexes..." -ForegroundColor Yellow
py scripts/build_index.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 4: Full Evaluation (BERTScore & LLM Judge)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [4/5] Running Model Evaluation..." -ForegroundColor Yellow
py scripts/run_eval.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 5: Retrieval Ablation Study (Dense vs Hybrid vs Rerank)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [5/5] Running Ablation Experiments..." -ForegroundColor Yellow
py scripts/run_ablation.py

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host " All Pipeline Stages Completed Successfully!" -ForegroundColor Green
Write-Host " Results saved in artifacts/" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
