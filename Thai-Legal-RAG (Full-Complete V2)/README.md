# ⚖️ Thai Legal RAG Q&A System

ระบบถาม-ตอบเอกสารกฎหมายไทยด้วยสถาปัตยกรรม Retrieval-Augmented Generation (RAG) ผสาน Hybrid Search (Dense Vector + BM25) ร่วมกับ Cross-Encoder Reranker พร้อมระบบ OCR สำรองสำหรับเอกสาร PDF ที่ Text Layer เสียหาย และชุดทดสอบ Benchmark กฎหมายมาตรฐานภายนอก (WangchanX, NitiBench)

---

## 📋 สารบัญ

- [1. การเตรียมระบบและติดตั้ง OCR Engine](#1-การเตรียมระบบและติดตั้ง-ocr-engine-system-dependencies)
- [2. การติดตั้ง Dependencies และสภาพแวดล้อม](#2-การติดตั้ง-dependencies-และสภาพแวดล้อม-python-environment)
- [3. การตั้งค่าระบบ](#3-การตั้งค่าระบบ-configuration)
- [4. ลำดับการรัน Pipeline แบบทีละขั้นตอน](#4-ลำดับการรัน-pipeline-แบบทีละขั้นตอน-step-by-step-execution)
- [5. การรัน Pipeline อัตโนมัติด้วยคำสั่งเดียว](#5-การรัน-pipeline-อัตโนมัติด้วยคำสั่งเดียว-one-click-pipeline)
- [6. การทดสอบ Benchmark กฎหมายมาตรฐานภายนอก](#6-การทดสอบ-benchmark-กฎหมายมาตรฐานภายนอก-external-benchmarks)
- [7. ผลลัพธ์และโครงสร้าง Artifacts](#7-ผลลัพธ์และโครงสร้าง-artifacts)

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

> 📌 **ก่อนเริ่ม**: ตรวจสอบให้แน่ใจว่า Terminal อยู่ในโฟลเดอร์โปรเจกต์ (root ของ repository) เรียบร้อยแล้ว

ทุกชุดคำสั่งด้านล่างทำงานครบ 4 ขั้นตอน: **(1)** กำหนด UTF-8 → **(2)** สร้าง Virtual Environment `.venv` → **(3)** เปิดใช้งาน (Activate) → **(4)** อัปเกรด pip และติดตั้งไลบรารีทั้งหมด (รวม `datasets` สำหรับชุด Benchmark ในหัวข้อที่ 6)

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
py -m pip install datasets
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
py -m pip install datasets
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
pip install datasets
```

> 💡 **หมายเหตุ**:
> - หากเครื่องใช้คำสั่ง `py` ให้พิมพ์ `py -m venv .venv` แทน `python -m venv .venv`
> - สร้าง `.venv` เพียงครั้งเดียว เมื่อปิด Terminal แล้วเปิดใหม่ **ไม่ต้องสร้างใหม่** ให้ทำเพียงตั้งค่า UTF-8 และ Activate อีกครั้ง (CMD: `.venv\Scripts\activate` / PowerShell: `.\.venv\Scripts\Activate.ps1` / Linux & macOS: `source .venv/bin/activate`)

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

ตรวจสอบความสอดคล้องของ Config, Metrics, Testset, Corpus, Index และ Retriever ก่อนเริ่มทดลองจริง

```bash
# Windows (CMD / PowerShell)
py scripts/smoke_test.py

# Linux / macOS
python3 scripts/smoke_test.py
```

ต้องผ่านการทดสอบขึ้นสถานะ ✅ / (PASS) ครบทุกหัวข้อ จึงถือว่าสภาพแวดล้อม `.venv` สมบูรณ์และพร้อมสำหรับรัน Pipeline หรือ Benchmark ต่อได้ทันที

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

## 6. การทดสอบ Benchmark กฎหมายมาตรฐานภายนอก (External Benchmarks)

ใช้ทดสอบระบบกับชุดข้อมูลกฎหมายไทยมาตรฐานภายนอก โดยต้องผ่านหัวข้อที่ 2 (ติดตั้ง `datasets` แล้ว) และ Smoke Test (หัวข้อ 4 ขั้นตอนที่ 3) มาก่อน

> ⚠️ **ลำดับสำคัญ**: ให้รัน **WangchanX ก่อน** เนื่องจาก NitiBench ใช้ Index ที่ WangchanX สร้างไว้ (`--use_wangchanx_corpus`)

### 6.1 WangchanX (กฎหมายแพ่งและพาณิชย์)

```bash
# Windows (CMD / PowerShell)
py scripts/benchmark_wangchanx.py --samples 30 --mode all

# Linux / macOS
python3 scripts/benchmark_wangchanx.py --samples 30 --mode all
```

- สร้าง Index ของคลังกฎหมายที่ `artifacts/wangchanx/` (`chunks.json`, `bm25.pkl`, `dense.faiss`)
- บันทึกผลที่ `artifacts/benchmarks/`: `wangchanx_benchmark_results.json`, `wangchanx_benchmark_table.md`, `wangchanx_results_detail.json`, `wangchanx_results_table.md`

### 6.2 NitiBench (ข้อสอบและการวินิจฉัยกฎหมาย)

```bash
# Windows (CMD / PowerShell)
py scripts/benchmark_nitibench.py --samples 30 --mode all --use_wangchanx_corpus

# Linux / macOS
python3 scripts/benchmark_nitibench.py --samples 30 --mode all --use_wangchanx_corpus
```

- ใช้ Index ประมวลกฎหมายแพ่งและพาณิชย์ (ป.พ.พ.) จาก `artifacts/wangchanx/` ในการค้นหาคำตอบ
- เก็บ `chunks.json` ของชุดทดสอบนี้ที่ `artifacts/nitibench/`
- บันทึกผลที่ `artifacts/benchmarks/`: `nitibench_results_detail.json`, `nitibench_results_table.md`

> 💡 `--samples 30` คือจำนวนตัวอย่างที่สุ่มมาทดสอบ และ `--mode all` คือรันทุกโหมดการค้นคืน ปรับค่าได้ตามต้องการ

---

## 7. ผลลัพธ์และโครงสร้าง Artifacts

หลังการรัน Pipeline ครบถ้วน ข้อมูลผลลัพธ์ทั้งหมดจะถูกจัดเก็บไว้ในโฟลเดอร์ `artifacts/` โดยมีโครงสร้างดังนี้:

```text
artifacts/
├── dense.faiss               # Index หลัก (Dense)
├── bm25.pkl                  # Index หลัก (BM25)
├── chunks.json               # Chunks ของคลังกฎหมายหลัก
├── ablation_table.md         # ผล Ablation Study (ตาราง)
├── eval_results.json         # ผลประเมิน End-to-End
├── eval_raw_rows.json        # ผลดิบรายข้อของการประเมิน
├── failure_cases.json        # เคสที่ล้มเหลว
├── wangchanx/                # Index ของคลัง WangchanX (ป.พ.พ.)
│   ├── dense.faiss
│   ├── bm25.pkl
│   └── chunks.json
├── nitibench/
│   └── chunks.json           # Chunks ของ NitiBench
└── benchmarks/               # ผล Benchmark มาตรฐานภายนอก
    ├── wangchanx_benchmark_results.json
    ├── wangchanx_benchmark_table.md
    ├── wangchanx_results_detail.json
    ├── wangchanx_results_table.md
    ├── nitibench_results_detail.json
    └── nitibench_results_table.md
```

### 7.1 Artifacts หลักของระบบ

| ไฟล์ Artifact | รายละเอียดของข้อมูล | การนำไปใช้งาน |
|---|---|---|
| `dense.faiss` | Vector Index สำหรับการค้นหาเชิงความหมาย (Bi-Encoder) | ใช้งานใน Retriever Mode Dense / Hybrid |
| `bm25.pkl` | Sparse Index สำหรับการค้นหาด้วยคีย์เวิร์ดเฉพาะทางกฎหมาย | ใช้งานใน Retriever Mode BM25 / Hybrid |
| `chunks.json` | ก้อนข้อความกฎหมายทั้งหมดหลังการทำ Token-aware Chunking | ใช้อ้างอิงและประกอบ Context ส่งเข้า LLM |
| `ablation_results.json` | ผลการทดลองเปรียบเทียบทั้ง 4 Retrieval Variants เชิงตัวเลข | ใช้ตรวจสอบสถิติละเอียด (Bootstrap CI, Latency) |
| `ablation_table.md` | ตารางสรุปผล Retrieval Metrics ในรูปแบบ Markdown | สำหรับนำไปวางใน Presentation Slides (หน้า 6–7) |
| `eval_results.json` | ผลการประเมิน BERTScore, LLM Judge และ Abstention Rate | สำหรับนำไปวางใน Presentation Slides (หน้า 8) |
| `eval_raw_rows.json` | ผลดิบรายข้อของการประเมิน (คำตอบและคะแนนรายคำถาม) | ใช้ตรวจสอบย้อนหลังและวิเคราะห์เชิงลึกรายข้อ |
| `failure_cases.json` | บันทึกเคสที่ระบบปฏิเสธผิดพลาดหรือดึงเอกสารไม่สมบูรณ์ | สำหรับนำไปใช้อภิปรายใน Failure Analysis (หน้า 9) |

### 7.2 Artifacts ของ Benchmark ภายนอก

| ไฟล์ / โฟลเดอร์ | รายละเอียดของข้อมูล | การนำไปใช้งาน |
|---|---|---|
| `wangchanx/` | Index (`dense.faiss`, `bm25.pkl`, `chunks.json`) ของคลังกฎหมายแพ่งและพาณิชย์ | ใช้ค้นหาใน WangchanX และ NitiBench (`--use_wangchanx_corpus`) |
| `nitibench/chunks.json` | Chunks ที่เกี่ยวข้องกับชุดทดสอบ NitiBench | ใช้อ้างอิงระหว่างรัน NitiBench |
| `benchmarks/wangchanx_benchmark_results.json`, `wangchanx_results_detail.json` | ผลสรุปและผลรายข้อของ WangchanX (JSON) | ตรวจสอบตัวเลขละเอียด |
| `benchmarks/wangchanx_benchmark_table.md`, `wangchanx_results_table.md` | ตารางสรุปผล WangchanX (Markdown) | นำไปวางในรายงาน / Slides |
| `benchmarks/nitibench_results_detail.json` | ผลรายข้อของ NitiBench (JSON) | ตรวจสอบตัวเลขละเอียด |
| `benchmarks/nitibench_results_table.md` | ตารางสรุปผล NitiBench (Markdown) | นำไปวางในรายงาน / Slides |
