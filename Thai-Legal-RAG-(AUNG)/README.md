# ⚖️ Thai Legal RAG Q&A System

ระบบถาม-ตอบเอกสารกฎหมายไทยด้วยเทคนิค RAG (Retrieval-Augmented Generation) พร้อมระบบ OCR สำรองสำหรับเอกสาร PDF ที่ Text Layer เสียหาย

---

## 📋 สารบัญ

- [1. การเตรียมระบบและติดตั้ง OCR Engine](#1-การเตรียมระบบและติดตั้ง-ocr-engine-system-dependencies)
- [2. การติดตั้ง Dependencies และสภาพแวดล้อม](#2-การติดตั้ง-dependencies-และสภาพแวดล้อม-python-environment)
- [3. การตั้งค่าระบบ](#3-การตั้งค่าระบบ-configuration)
- [4. ลำดับการรัน Pipeline](#4-ลำดับการรัน-pipeline-แบบทีละขั้นตอน-step-by-step-execution)
- [5. การรัน Pipeline อัตโนมัติด้วยคำสั่งเดียว](#5-การรัน-pipeline-อัตโนมัติด้วยคำสั่งเดียว-one-click-pipeline)

---

## 1. การเตรียมระบบและติดตั้ง OCR Engine (System Dependencies)

โปรเจกต์นี้มีระบบ **Fallback อัตโนมัติไปยัง Tesseract OCR** สำหรับเอกสาร PDF กฎหมายที่ Text Layer เสียหาย จำเป็นต้องติดตั้ง Tesseract พร้อมรองรับภาษาไทย (`tha`) ตามระบบปฏิบัติการดังนี้

### 🪟 Windows

1. ดาวน์โหลดตัวติดตั้ง (`.exe`) สำหรับ 64-bit จาก [UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. เปิดตัวติดตั้ง กด **Next** จนถึงหน้าต่าง **Choose Components**
3. คลี่หัวข้อ **Additional script data** และ **Additional language data** แล้วติ๊กเลือก **Thai**
4. ตรวจสอบให้แน่ใจว่าติดตั้งไว้ที่ Path เริ่มต้น (`C:\Program Files\Tesseract-OCR`)

### 🐧 Ubuntu / Debian / WSL

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-tha
```

### 🍎 macOS (Apple Silicon / Intel)

```bash
brew install tesseract tesseract-lang
```

---

## 2. การติดตั้ง Dependencies และสภาพแวดล้อม (Python Environment)

แนะนำให้ใช้ **Python 3.10+** และตั้งค่า Virtual Environment ก่อนติดตั้งไลบรารี

### 🪟 Windows (PowerShell)

```powershell
# บังคับใช้ UTF-8 เพื่อป้องกันข้อผิดพลาดจากภาษาไทย
$env:PYTHONUTF8=1

# สร้างและเปิดใช้งาน venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# ติดตั้งไลบรารี
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 🐧🍎 Linux & macOS (Bash / Zsh)

```bash
# บังคับใช้ UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# สร้างและเปิดใช้งาน venv
python3 -m venv .venv
source .venv/bin/activate

# ติดตั้งไลบรารี
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. การตั้งค่าระบบ (Configuration)

ตรวจสอบไฟล์ `config.py` และระบุข้อมูล Endpoint และ Token สำหรับ LLM Proxy / OpenAI API:

```python
LLM_BASE_URL = "https://llm.nat-d.uk/v1"  # หรือ OpenAI API endpoint
LLM_API_KEY = "YOUR_API_KEY"
LLM_MODEL = "gemma-4-E4B-it"             # หรือโมเดลที่ต้องการใช้งาน
JUDGE_MODEL = "gemma-4-E4B-it"
```

> ⚠️ **หมายเหตุ:** อย่า commit ไฟล์ที่มี API Key จริงขึ้น Git — แนะนำให้ใช้ตัวแปรแวดล้อม (environment variable) หรือไฟล์ `.env` ที่อยู่ใน `.gitignore`

---

## 4. ลำดับการรัน Pipeline แบบทีละขั้นตอน (Step-by-Step Execution)

### ขั้นตอนที่ 1 — สกัดข้อความจาก PDF (Data Ingestion)

แปลงไฟล์ PDF กฎหมายทั้งหมดให้เป็นข้อความที่สะอาด พร้อมบันทึกรายงานสถิติ extraction

```bash
# Windows
py src/extract_corpus.py

# Linux / macOS
python3 src/extract_corpus.py
```

### ขั้นตอนที่ 2 — แบ่งท่อนข้อความและสร้าง Index (Build Indexes)

สร้าง Vector Index (Dense Embedding ด้วย `multilingual-e5-base`) และ Sparse Index (BM25)

```bash
# Windows
py scripts/build_index.py

# Linux / macOS
python3 scripts/build_index.py
```

### ขั้นตอนที่ 3 — ทดสอบระบบถาม-ตอบ (Interactive Demo)

รันเดโมเพื่อทดลองป้อนคำถามกฎหมายและตรวจสอบเอกสารอ้างอิง

```bash
# Windows
py demo.py

# Linux / macOS
python3 demo.py
```

### ขั้นตอนที่ 4 — การประเมินผลและการทดลอง (Evaluation & Ablation)

**รันประเมินความแม่นยำ** (BERTScore & LLM Judge):

```bash
py scripts/run_eval.py          # Windows
python3 scripts/run_eval.py     # Linux / macOS
```

**รันการทดลองเปรียบเทียบ Retrieval** (Ablation Study: Dense vs Hybrid vs Reranker):

```bash
py scripts/run_ablation.py      # Windows
python3 scripts/run_ablation.py # Linux / macOS
```

---

## 5. การรัน Pipeline อัตโนมัติด้วยคำสั่งเดียว (One-Click Pipeline)

### 🪟 Windows

```powershell
.\run_pipeline.ps1
```

### 🐧🍎 Linux / macOS

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

<p align="center">
Made with ⚖️ + 🧠 for Thai legal document Q&A
</p>