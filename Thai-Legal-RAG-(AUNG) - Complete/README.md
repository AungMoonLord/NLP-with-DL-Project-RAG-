# ⚖️ Thai Legal RAG Q&A System

ระบบถาม-ตอบเอกสารกฎหมายไทยด้วยสถาปัตยกรรม Retrieval-Augmented Generation (RAG) ผสาน Hybrid Search (Dense Vector + BM25) ร่วมกับ Cross-Encoder Reranker พร้อมระบบ OCR สำรองสำหรับเอกสาร PDF ที่ Text Layer เสียหาย

---

## 📋 สารบัญ

- [1. การเตรียมระบบและติดตั้ง OCR Engine](#1-การเตรียมระบบและติดตั้ง-ocr-engine-system-dependencies)
- [2. การติดตั้ง Dependencies และสภาพแวดล้อม](#2-การติดตั้ง-dependencies-และสภาพแวดล้อม-python-environment)
- [3. การตั้งค่าระบบ](#3-การตั้งค่าระบบ-configuration)
- [4. ลำดับการรัน Pipeline แบบทีละขั้นตอน](#4-ลำดับการรัน-pipeline-แบบทีละขั้นตอน-step-by-step-execution)
- [5. การรัน Pipeline อัตโนมัติด้วยคำสั่งเดียว](#5-การรัน-pipeline-อัตโนมัติด้วยคำสั่งเดียว-one-click-pipeline)
- [6. ผลลัพธ์และโครงสร้าง Artifacts](#6-ผลลัพธ์และโครงสร้าง-artifacts)

---

## 1. การเตรียมระบบและติดตั้ง OCR Engine (System Dependencies)

โปรเจกต์นี้มีระบบ **Fallback อัตโนมัติไปยัง Tesseract OCR** สำหรับเอกสาร PDF กฎหมายที่ไม่มี Text Layer หรือข้อความเสียหาย จำเป็นต้องติดตั้ง Tesseract พร้อมข้อมูลภาษาไทย (`tha`) ตามระบบปฏิบัติการดังนี้

### 🪟 Windows

1. ดาวน์โหลดตัวติดตั้ง (`.exe`) สำหรับ 64-bit จาก [UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. เปิดตัวติดตั้ง กด **Next** จนถึงหน้าต่าง **Choose Components**
3. คลี่หัวข้อ **Additional script data** และ **Additional language data** แล้วติ๊กเลือก **Thai**
4. ตรวจสอบให้แน่ใจว่าติดตั้งไว้ที่ Path เริ่มต้น (`C:\Program Files\Tesseract-OCR\tesseract.exe`)

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

แนะนำให้ใช้ Python 3.10+ โดยแยกการตั้งค่าสภาพแวดล้อมตาม Terminal ที่ใช้งาน

### 🪟 Windows (Command Prompt - CMD)

```dos
:: บังคับใช้ UTF-8 เพื่อป้องกันข้อผิดพลาดจากภาษาไทยและสระลอย
chcp 65001
set PYTHONUTF8=1

:: สร้างและเปิดใช้งาน Virtual Environment
py -m venv .venv
call .venv\Scripts\activate.bat

:: อัปเกรด pip และติดตั้งไลบรารี
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 🪟 Windows (PowerShell)

```powershell
# บังคับใช้ UTF-8
$env:PYTHONUTF8=1

# สร้างและเปิดใช้งาน Virtual Environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# อัปเกรด pip และติดตั้งไลบรารี
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 🐧🍎 Linux & macOS (Bash / Zsh)

```bash
# บังคับใช้ UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# สร้างและเปิดใช้งาน Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# อัปเกรด pip และติดตั้งไลบรารี
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. การตั้งค่าระบบ (Configuration)

ตรวจสอบไฟล์ `config.py` เพื่อระบุค่า Endpoint และ Key สำหรับโมเดลภาษา:

```python
LLM_BASE_URL = "https://llm.nat-d.uk/v1"  # หรือคลาสเรียน Endpoint / OpenAI API
LLM_API_KEY = "YOUR_API_KEY"
LLM_MODEL = "gemma-4-E4B-it"             # โมเดล Generator หลัก
JUDGE_MODEL = "gemma-4-E4B-it"           # โมเดล LLM-as-Judge
```

> ⚠️ **หมายเหตุ**: หลีกเลี่ยงการ Commit ไฟล์ที่มี API Key ขึ้น Git โดยเด็ดขาด

---

## 4. ลำดับการรัน Pipeline แบบทีละขั้นตอน (Step-by-Step Execution)

### ขั้นตอนที่ 1 — สกัดข้อความจาก PDF (Data Ingestion)

แปลงไฟล์ PDF ข้อกฎหมายใน `data/pdfs/` ให้เป็นข้อความที่ผ่านการ Clean ใน `data/documents/` พร้อมระบบ OCR สำรอง

```bash
# Windows (CMD / PowerShell)
py src/extract_corpus.py

# Linux / macOS
python3 src/extract_corpus.py
```

### ขั้นตอนที่ 2 — แบ่งท่อนข้อความและสร้าง Index (Build Indexes)

ตัดข้อความแบบ Token-aware Chunks (~256 tokens) และสร้าง Dense Index (multilingual-e5-base) ควบคู่กับ Sparse Index (BM25)

```bash
# Windows (CMD / PowerShell)
py scripts/build_index.py

# Linux / macOS
python3 scripts/build_index.py
```

### ขั้นตอนที่ 3 — ตรวจสอบความพร้อมของระบบ (Smoke Test)

ตรวจสอบความสอดคล้องของ Config, Metrics, Testset, Corpus และ Retriever ก่อนเริ่มทดลองจริง

```bash
# Windows (CMD / PowerShell)
py scripts/smoke_test.py

# Linux / macOS
python3 scripts/smoke_test.py
```

ต้องผ่านการทดสอบขึ้นเครื่องหมาย ✅ ครบทุกหัวข้อ

### ขั้นตอนที่ 4 — รันการทดลองเปรียบเทียบ Retrieval (Ablation Study)

วัดประสิทธิภาพเปรียบเทียบ 4 Variants (dense, bm25, hybrid, hybrid_rerank) บน Metrics: Recall@k, MRR@k, nDCG@k, Latency และค่าสถิติ Bootstrap CI

```bash
# Windows (CMD / PowerShell)
py scripts/run_ablation.py

# Linux / macOS
python3 scripts/run_ablation.py
```

### ขั้นตอนที่ 5 — ประเมินผลระบบถาม-ตอบแบบครบวงจร (Full Evaluation)

รัน End-to-End Pipeline เพื่อวัด Semantic Similarity (BERTScore xlm-roberta-large), LLM-as-Judge (Faithfulness & Relevance), อัตราการปฏิเสธคำตอบที่อยู่นอกขอบเขต (Abstention Rate) และสกัดเคสล้มเหลวอัตโนมัติ

```bash
# Windows (CMD / PowerShell)
py scripts/run_eval.py

# Linux / macOS
python3 scripts/run_eval.py
```

### ขั้นตอนที่ 6 — ทดสอบระบบถาม-ตอบสด (Interactive Live Demo)

รันหน้าจอโต้ตอบสดแบบ Streaming (พิมพ์ดีดทีละ Token) พร้อมแสดงเอกสารและมาตราอ้างอิง

```bash
# Windows (CMD / PowerShell)
py demo.py

# Linux / macOS
python3 demo.py
```

(พิมพ์คำถามกฎหมายที่ต้องการ หรือพิมพ์ `exit` เพื่อออกจากระบบ)

---

## 5. การรัน Pipeline อัตโนมัติด้วยคำสั่งเดียว (One-Click Pipeline)

### 🪟 Windows (PowerShell)

```powershell
.\run_pipeline.ps1
```

### 🐧🍎 Linux / macOS (Bash)

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 6. ผลลัพธ์และโครงสร้าง Artifacts

หลังการรัน Pipeline ครบถ้วน ข้อมูลผลลัพธ์ทั้งหมดจะถูกจัดเก็บไว้ในโฟลเดอร์ `artifacts/` ดังนี้:

| ไฟล์ Artifact | รายละเอียดของข้อมูล | การนำไปใช้งาน |
|---|---|---|
| `dense.faiss` | Vector Index สำหรับการค้นหาเชิงความหมาย (Bi-Encoder) | ใช้งานใน Retriever Mode Dense / Hybrid |
| `bm25.pkl` | Sparse Index สำหรับการค้นหาด้วยคีย์เวิร์ดเฉพาะทางกฎหมาย | ใช้งานใน Retriever Mode BM25 / Hybrid |
| `chunks.json` | ก้อนข้อความกฎหมายทั้งหมดหลังการทำ Token-aware Chunking | ใช้อ้างอิงและประกอบ Context ส่งเข้า LLM |
| `ablation_results.json` | ผลการทดลองเปรียบเทียบทั้ง 4 Retrieval Variants เชิงตัวเลข | ใช้ตรวจสอบสถิติละเอียด (Bootstrap CI, Latency) |
| `ablation_table.md` | ตารางสรุปผล Retrieval Metrics ในรูปแบบ Markdown | สำหรับนำไปวางใน Presentation Slides (หน้า 6–7) |
| `eval_results.json` | ผลการประเมิน BERTScore, LLM Judge และ Abstention Rate | สำหรับนำไปวางใน Presentation Slides (หน้า 8) |
| `failure_cases.json` | บันทึกเคสที่ระบบปฏิเสธผิดพลาดหรือดึงเอกสารไม่สมบูรณ์ | สำหรับนำไปใช้อภิปรายใน Failure Analysis (หน้า 9) |
