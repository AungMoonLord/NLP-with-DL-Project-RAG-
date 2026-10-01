#!/usr/bin/env bash
# ==============================================================================
# Pipeline Execution Script for thai-legal-rag
# Supports: macOS & Linux / Ubuntu / WSL
# ==============================================================================

set -e # หยุดการทำงานทันทีหากคำสั่งใดเกิด error

# กำหนด Encoding เป็น UTF-8 ป้องกันสระภาษาไทยเพี้ยน
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "=================================================="
echo " Starting Full Legal-RAG Pipeline"
echo "=================================================="

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 0: ติดตั้ง/ตรวจสอบ Tesseract OCR ภาษาไทย
# ------------------------------------------------------------------------------
echo -e "\n>>> [0/5] Checking OCR System Dependencies..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v tesseract &> /dev/null; then
        echo "Tesseract not found. Installing via Homebrew..."
        brew install tesseract tesseract-lang
    else
        echo "Tesseract found: $(which tesseract)"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux / Ubuntu / Debian / WSL
    if ! command -v tesseract &> /dev/null; then
        echo "Tesseract not found. Installing via apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-tha
    else
        echo "Tesseract found: $(which tesseract)"
    fi
fi

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 1: ตั้งค่า Python Virtual Environment & Dependencies
# ------------------------------------------------------------------------------
echo -e "\n>>> [1/5] Setting up Python Environment..."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 2: Data Extraction (PDF -> Clean Text)
# ------------------------------------------------------------------------------
echo -e "\n>>> [2/5] Extracting & Cleaning Text from Legal PDFs..."
python3 src/extract_corpus.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 3: Chunking & Indexing (FAISS + BM25)
# ------------------------------------------------------------------------------
echo -e "\n>>> [3/5] Building Vector & BM25 Indexes..."
python3 scripts/build_index.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 4: Full Evaluation (BERTScore & LLM Judge)
# ------------------------------------------------------------------------------
echo -e "\n>>> [4/5] Running Model Evaluation..."
python3 scripts/run_eval.py

# ------------------------------------------------------------------------------
# ขั้นตอนที่ 5: Retrieval Ablation Study (Dense vs Hybrid vs Rerank)
# ------------------------------------------------------------------------------
echo -e "\n>>> [5/5] Running Ablation Experiments..."
python3 scripts/run_ablation.py

echo -e "\n=================================================="
echo " All Pipeline Stages Completed Successfully!"
echo " Results saved in artifacts/"
echo "=================================================="

