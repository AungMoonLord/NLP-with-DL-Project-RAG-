# Thai Legal RAG — ระบบถาม-ตอบกฎหมายไทย

ระบบ Q&A เฉพาะโดเมนกฎหมายไทย ทำงานบนคลังเอกสารจริง 85 ฉบับ
(ข้อบังคับการประชุมรัฐสภา, พระราชบัญญัติ, พระราชบัญญัติประกอบรัฐธรรมนูญ, รัฐธรรมนูญ)

ค้นคืนแบบสองขั้น: dense + BM25 รวมด้วย RRF แล้วจัดอันดับซ้ำด้วย cross-encoder
ตอบเป็นภาษาไทยพร้อมอ้างอิงข้อ/มาตราที่ตรวจสอบย้อนกลับได้

---

## สารบัญ

1. [สิ่งที่ต้องมีก่อน](#1-สิ่งที่ต้องมีก่อน)
2. [ติดตั้ง 6 ขั้น](#2-ติดตั้ง-6-ขั้น)
3. [เปิดหน้าเว็บ](#3-เปิดหน้าเว็บ)
4. [คำสั่งทั้งหมด](#4-คำสั่งทั้งหมด)
5. [ปัญหาที่เจอบ่อยและวิธีแก้](#5-ปัญหาที่เจอบ่อยและวิธีแก้)
6. [ถ้าแตกไฟล์ zip ทับโฟลเดอร์เดิม](#-ถ้าแตกไฟล์-zip-ทับโฟลเดอร์เดิม)
7. [สถาปัตยกรรมระบบ](#6-สถาปัตยกรรมระบบ)
7. [การตัดสินใจเชิงเทคนิค](#7-การตัดสินใจเชิงเทคนิค)
8. [โครงสร้างไฟล์](#8-โครงสร้างไฟล์)

---

## 1. สิ่งที่ต้องมีก่อน

| สิ่งที่ต้องมี | รายละเอียด |
|---|---|
| Python | 3.10 ขึ้นไป |
| Node.js | 18 ขึ้นไป (เฉพาะถ้าจะใช้หน้าเว็บ) จาก nodejs.org เลือกรุ่น LTS |
| พื้นที่ว่าง | ~6 GB (โมเดล 3.5 GB + index 25 MB + ไลบรารี) |
| RAM | 8 GB ขึ้นไป |
| API key | endpoint ที่รองรับ OpenAI API เช่น OpenTyphoon, OpenRouter หรือของมหาวิทยาลัย |

**สำคัญ: คลังเอกสารไม่ได้อยู่ใน repository** เพราะขนาดใหญ่และติดเรื่องลิขสิทธิ์
ต้องขอไฟล์จากทีม แล้ววางใน `data/raw/` เอง (รองรับ `.txt` `.md` `.pdf` `.json` `.jsonl`)

ระบบทั้งหมดรันบน CPU ได้ ไม่ต้องใช้ GPU แต่ขั้นสร้าง index ใช้เวลาประมาณ 30 นาที

---

## 2. ติดตั้ง 6 ขั้น

### ขั้น 1 — สร้าง virtual environment และติดตั้งไลบรารี

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

ใช้เวลา 5-15 นาที เพราะต้องโหลด PyTorch ซึ่งไฟล์ใหญ่

### ขั้น 2 — ตรวจว่าติดตั้งสำเร็จ

```bash
python tests/test_offline.py
```

ต้องขึ้น "ผ่านทุกการทดสอบ" ชุดทดสอบนี้ไม่ใช้โมเดลและไม่ใช้เน็ต
ตรวจ chunker, RRF, เมตริกค้นคืน และการตัดคำไทย

ถ้า ROUGE-L ได้ค่า 0.0 แปลว่า pythainlp ยังไม่ทำงาน ให้ติดตั้งใหม่

### ขั้น 3 — ตั้งค่า API key

สร้างไฟล์ชื่อ `.env` ที่โฟลเดอร์หลัก (ระดับเดียวกับ README นี้) ใส่:

```
OPENAI_API_KEY=คีย์ของคุณ
OPENAI_BASE_URL=https://endpoint-ของคุณ/v1
```

> Windows: Notepad จะเติม `.txt` ต่อท้ายให้เอง ตอนบันทึกต้องเลือก
> "Save as type: All Files" แล้วพิมพ์ชื่อ `.env` ให้ครบ
> ห้าม commit ไฟล์นี้ขึ้น git (มีใน `.gitignore` แล้ว)

ดูชื่อ model ที่ endpoint รองรับ:

```bash
python scripts/test_api.py --list
```

แล้วใส่ชื่อลงไฟล์ config ทั้ง 6 ไฟล์:

```bash
python -c "from pathlib import Path; [p.write_text(p.read_text(encoding='utf-8').replace('MODEL_NAME_HERE','ชื่อ-model-ของคุณ'),encoding='utf-8') for p in Path('configs').glob('*.yaml')]"
```

ทดสอบการเชื่อมต่อ:

```bash
python scripts/test_api.py
```

### ขั้น 4 — วางคลังเอกสาร

วางไฟล์ทั้งหมดใน `data/raw/` แล้วตรวจว่าอ่านได้ครบ:

```bash
python scripts/ingest.py --stats-only
```

ดูว่า `n_documents` ตรงกับจำนวนไฟล์จริง ถ้าน้อยกว่าแปลว่ามีไฟล์อ่านไม่ออก
(ระบบจะพิมพ์ชื่อไฟล์ที่ข้ามให้) PDF สแกนต้อง OCR ก่อน

<details>
<summary>ถ้าคลังมี PDF ที่เป็นภาพสแกน (OCR)</summary>

ตรวจก่อนว่ามีปัญหาจริงแค่ไหน:

```bash
python scripts/check_scanned.py
```

สคริปต์นี้นับตัวอักษรที่ดึงได้ต่อหน้า หน้าข้อความปกติของกฎหมายไทยจะได้
800-2,500 ตัวอักษร ส่วนหน้าที่เป็นภาพสแกนจะได้เกือบ 0

- **น้อยกว่า 5%** ของหน้าทั้งหมด แนะนำให้บันทึกเป็นข้อจำกัดของระบบ
  แล้วเอาเวลาไปทำ ablation กับ evaluation ซึ่งมีคะแนนมากกว่า
- **มากกว่า 5%** ควรทำ OCR เฉพาะไฟล์ที่มีปัญหา

วิธีทำ OCR (Windows):

1. ติดตั้ง Tesseract จาก https://github.com/UB-Mannheim/tesseract/wiki
   ตอนติดตั้งต้องติ๊กเลือก **Thai** ในหน้า Additional language data
2. `pip install ocrmypdf`
3. รันเฉพาะไฟล์ที่มีปัญหา:

```bash
ocrmypdf -l tha --skip-text ไฟล์เดิม.pdf ไฟล์ใหม่.pdf
```

`--skip-text` สำคัญมาก: มันจะ OCR เฉพาะหน้าที่ไม่มีข้อความ
และไม่ไปทับข้อความเดิมที่ดีอยู่แล้ว

⚠️ **ข้อควรรู้ก่อนตัดสินใจ:** OCR ภาษาไทยมีความแม่นยำต่ำกว่าภาษาอังกฤษมาก
โดยเฉพาะวรรณยุกต์และสระบน-ล่าง ผลที่ได้จะมีความเสียหายแบบเดียวกับที่
`cleaner.py` พยายามแก้อยู่ (เช่น "กาหนด" แทน "กำหนด") แต่หนักกว่า
ควรสุ่มอ่านผล OCR ด้วยตาก่อนเอาเข้าคลัง

</details>

### ขั้น 5 — ทำความสะอาดข้อความ

ข้ามได้ถ้าคลังไม่ได้มาจากการแปลง PDF แต่ถ้ามาจาก PDF ควรทำ

```bash
python scripts/clean_corpus.py --dry-run --skip-suggest   # ดูก่อนว่าจะแก้อะไร
```

เปิดอ่าน `data/processed/cleaning_report.md`
**ถ้าขนาดลดลงเกิน 15% ให้สงสัยว่าลบเนื้อหาจริง** แก้ด้วย `--min-run 5` แล้วรันใหม่

พอใจแล้วรันจริง:

```bash
python scripts/clean_corpus.py --skip-suggest
```

ผลไปอยู่ที่ `data/clean/` ของเดิมใน `data/raw/` ไม่ถูกแตะ
จากนั้นบอกระบบให้ใช้ไฟล์สะอาด:

```bash
python -c "from pathlib import Path; [p.write_text('raw_dir: data/clean\n'+p.read_text(encoding='utf-8'),encoding='utf-8') for p in Path('configs').glob('*.yaml') if 'raw_dir' not in p.read_text(encoding='utf-8')]"
```

<details>
<summary>ตัวเลือกเสริม: ให้ระบบหาคำที่สระ/วรรณยุกต์หายจาก PDF</summary>

```bash
python scripts/clean_corpus.py --dry-run     # ไม่ใส่ --skip-suggest (ช้ากว่ามาก)
```

ผลไปที่ `data/processed/corrections_candidates.tsv` เป็นรายการเสนอ เช่น
`กาหนด → กำหนด`, `ขอบังคับ → ข้อบังคับ`

**ระบบจะไม่แก้ให้เอง** ต้องอ่านทุกบรรทัด ลบที่ผิดทิ้ง แล้ว copy ที่เหลือ
ไปต่อท้าย `data/corrections.txt` เหตุผล: `ขอบังคับ` เป็นคำจริงได้ (ขอ + บังคับ)
พจนานุกรมบอกได้แค่ว่า "คำนี้มีอยู่" ไม่ได้บอกว่า "ในบริบทนี้ควรเป็นคำไหน"
การแก้ตัวบทกฎหมายผิดคือความเสียหายที่มองไม่เห็น
</details>

### ขั้น 6 — สร้าง index

```bash
python scripts/ingest.py
```

**ใช้เวลาประมาณ 30 นาทีบน CPU** ครั้งแรกจะโหลดโมเดล 1.1 GB ด้วย

ตรวจบรรทัดนี้ก่อนปล่อยให้รันต่อ:

```
[chunker] token รวม header (ที่โมเดลเห็นจริง): max 500 | เกิน 512: 0 chunk  ✅
```

ถ้าขึ้น ⚠️ แปลว่ามี chunk ที่จะถูกตัดท้ายทิ้งเงียบ ๆ ให้หยุดแล้วแจ้งทีม

---

## 3. เปิดหน้าเว็บ

ต้องเปิด **สอง terminal แยกกัน**

**Terminal 1 — backend**

```bash
.venv\Scripts\activate
python -m uvicorn api.main:app --port 8000
```

รอจนขึ้น `[api] พร้อมใช้งานใน ... วินาที` (โหลดโมเดล 1-2 นาที) แล้วเปิดค้างไว้

**Terminal 2 — frontend**

```bash
cd frontend
npm install      # ครั้งแรกครั้งเดียว
npm run dev
```

เปิดเบราว์เซอร์ที่ **http://localhost:5173**

> ทุกครั้งที่รัน `ingest.py` ใหม่ ต้องปิด-เปิด uvicorn ใหม่ด้วย
> ไม่งั้นหน้าเว็บจะยังใช้ index เก่าที่ค้างในหน่วยความจำ

### สิ่งที่ทำได้บนหน้าเว็บ

- สลับวิธีค้นระหว่าง dense / BM25 / ผสม และเปิด-ปิด reranker **ได้ทันที** ใช้โชว์ ablation สด
- คลิก `[S1]` ในคำตอบ เพื่อกระโดดไปดูแหล่งอ้างอิงนั้น ใช้ตรวจ hallucination
- ป้ายบอกว่าแต่ละ chunk ถูกพบด้วย dense หรือ BM25 หรือทั้งคู่
- ตัวเลข `14 ▸ 5` บอกว่า reranker ดันอันดับขึ้นเท่าไร

### ถ้าไม่อยากใช้หน้าเว็บ

```bash
python scripts/demo.py --interactive --explain
```

ข้อเสีย: ต้องโหลดโมเดลใหม่ทุกครั้งที่เปิดโปรแกรม (~45 วินาทีต่อคำถาม)
ส่วน web server โหลดครั้งเดียวแล้วค้างไว้ คำถามถัดไปเหลือไม่กี่วินาที

---

## 4. คำสั่งทั้งหมด

```bash
# ตรวจระบบ
python tests/test_offline.py                    # ทดสอบ logic (ไม่ใช้เน็ต ไม่ใช้โมเดล)
python scripts/test_api.py                      # ทดสอบการเชื่อมต่อ LLM
python scripts/test_api.py --list               # ดู model ที่ endpoint รองรับ

# เตรียมข้อมูล
python scripts/ingest.py --stats-only           # ดูสถิติคลัง ไม่สร้าง index
python scripts/check_scanned.py                 # หาหน้าที่เป็นภาพสแกน (ต้อง OCR)
python scripts/text_quality.py --compare        # วัดความเสียหายของข้อความ ก่อน vs หลังทำความสะอาด
python scripts/compare_extractors.py            # เทียบ pypdf vs PyMuPDF บนคลังจริง
python scripts/clean_corpus.py --dry-run        # ดูว่าจะทำความสะอาดอะไร
python scripts/clean_corpus.py --skip-suggest   # ทำความสะอาดจริง (ข้ามขั้นที่ช้า)
python scripts/ingest.py                        # สร้าง chunk + dense + BM25 index
python scripts/ingest.py --skip-dense --skip-bm25   # ทดสอบ chunking อย่างเดียว (เร็ว)

# ใช้งาน
python scripts/demo.py                          # 3 คำถามตัวอย่าง
python scripts/demo.py --interactive --explain  # ถาม-ตอบสด พร้อมดู trace
python -m uvicorn api.main:app --port 8000      # เปิด API

# ประเมินผล
python eval/make_gold_set.py --n 40 --out data/qa_gold.draft.jsonl   # ถ้ายังไม่มีชุดทดสอบ
python eval/run_eval.py                         # ประเมิน 1 variant
python eval/run_eval.py --no-generate           # วัดเฉพาะขั้นค้นคืน (เร็วกว่ามาก)
python eval/run_ablation.py                     # เทียบทุก variant + ตาราง + กราฟ
python eval/failure_analysis.py --results results/eval_C.json
```

ทุกคำสั่งรับ `--config configs/xxx.yaml` ได้ ถ้าไม่ระบุจะใช้ `configs/baseline.yaml`

---

## 5. ปัญหาที่เจอบ่อยและวิธีแก้

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `pip install` พังตอนอ่าน requirements.txt | pip บน Windows อ่านไฟล์ด้วย locale cp874 ถ้ามีตัวอักษรไทยจะพัง | ไฟล์ปัจจุบันเป็น ASCII ล้วนแล้ว ถ้าแก้ไฟล์เอง อย่าใส่ภาษาไทย |
| `ModuleNotFoundError: No module named 'openai'` | ไม่ได้ติดตั้ง (อยู่ใน requirements เป็นคอมเมนต์) | `pip install openai` |
| `UnicodeEncodeError` ตอนรัน | Windows console ตั้งต้นเป็น cp874 | พิมพ์ `$env:PYTHONUTF8=1` ก่อนรัน |
| `TypeError: got an unexpected keyword argument 'temperature'` | config ยังเป็น backend anthropic | ตรวจว่า `configs/baseline.yaml` มี `backend: openai` |
| ขึ้น "ยังไม่ได้ตั้งชื่อ model" | ยังไม่ได้แทนที่ `MODEL_NAME_HERE` | ทำตามขั้น 3 |
| แก้ config แล้วไม่มีอะไรเปลี่ยน | รันโดยไม่ใส่ `--config` และแก้ไฟล์ที่ไม่ใช่ `baseline.yaml` | ระบบใช้ `baseline.yaml` เป็นค่าเริ่มต้น |
| `FileNotFoundError: chunks_*.jsonl` | ยังไม่ได้รัน ingest หรือเปลี่ยน chunk size แล้วยังไม่ ingest ใหม่ | รัน `python scripts/ingest.py` |
| หน้าเว็บขึ้น "เชื่อมต่อ backend ไม่ได้" | uvicorn ไม่ได้รัน หรือใช้ port อื่น | ตรวจ Terminal 1 ว่ายังทำงานอยู่ |
| ตัวเลข chunk บนหน้าเว็บไม่ตรงกับที่เพิ่ง ingest | index เก่าค้างในหน่วยความจำ | ปิด-เปิด uvicorn ใหม่ |
| คำถามแรกช้า 45 วินาที | โหลด cross-encoder 2.3 GB จากดิสก์ | ปกติ คำถามถัดไปจะเร็ว |
| ระบบกลับไปอ่าน `data/raw` เอง หรือหา model ไม่เจอ | แตก zip ทับ ทำให้ `configs/*.yaml` ถูกเขียนทับ | ดูหัวข้อถัดไป |

> ชื่อไฟล์ index ผูกกับ `(embedding model + chunk size + overlap)` เช่น
> `chunks_multilingual-e5-base_c512_o64.jsonl` ดังนั้น variant ที่ใช้ chunk คนละขนาด
> ต้อง ingest แยก และจะไม่เขียนทับกัน — กัน bug คลาสสิกที่เปลี่ยน chunk size
> แล้วลืมสร้าง index ใหม่

---

## ⚠️ ถ้าแตกไฟล์ zip ทับโฟลเดอร์เดิม

การแตก zip ทับจะ **เขียนทับไฟล์ `configs/*.yaml` ที่แก้ไว้แล้ว** ชื่อ model และ
`raw_dir` จะหายกลับไปเป็นค่าตั้งต้น ระบบจะไปอ่าน `data/raw` และหา model ไม่เจอ

รันสองคำสั่งนี้ทุกครั้งหลังแตก zip:

```bash
python -c "from pathlib import Path; [p.write_text(p.read_text(encoding='utf-8').replace('MODEL_NAME_HERE','ชื่อ-model-ของคุณ'),encoding='utf-8') for p in Path('configs').glob('*.yaml')]"
```

```bash
python -c "from pathlib import Path; [p.write_text('raw_dir: data/clean\n'+p.read_text(encoding='utf-8'),encoding='utf-8') for p in Path('configs').glob('*.yaml') if 'raw_dir' not in p.read_text(encoding='utf-8')]"
```

ตรวจว่าถูกต้อง:

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.config import Config; c=Config.load(); print('raw_dir =',c.raw_dir,'| model =',c.generator.model)"
```

สิ่งที่ **ไม่** ถูกเขียนทับ เพราะไม่ได้อยู่ใน zip: `.venv/`, `data/raw/`, `data/clean/`,
`data/processed/` (index), และ `.env`

---

## 6. สถาปัตยกรรมระบบ

```
data/raw/  85 ฉบับ
    |
    +--[1] Loader ---------- normalize + กรองเอกสารที่อ่านไม่ออก
    |
    +--[1.5] Cleaner ------- ลบหมายเหตุริมกระดาษ + ต่อบรรทัดที่ถูกตัด
    |                        + ลบช่องว่างแทรกกลางคำ (ตัดสินด้วยพจนานุกรม)
    |
    +--[2] ThaiLegalChunker  L1 ตัดตามขอบเขต "มาตรา"/"ข้อ"
    |                        L2 ซอยด้วยประโยค + overlap (นับ token จริง)
    |                        L3 รวมมาตราสั้น + แปะ breadcrumb
    |                        -> 7,093 chunks
    |
    +--[3a] Dense index ---- multilingual-e5-base, normalized, cosine
    +--[3b] BM25 index ----- ตัดคำด้วย newmm + แปลงเลขไทยเป็นอารบิก

คำถาม
    |
    +--[4] ขั้นที่ 1 -- dense top-20 + BM25 top-20 -> รวมด้วย RRF (k=60)
    |
    +--[5] ขั้นที่ 2 -- cross-encoder ให้คะแนนคู่ (คำถาม, chunk) -> เลือก top-5
    |
    +--[6] ประกอบ context + prompt ที่บังคับอ้างอิง [S#] และให้ปฏิเสธได้
    |
    +--> คำตอบภาษาไทย + แหล่งอ้างอิงที่ตรวจย้อนได้

[7] Evaluation -- ค้นคืน: Recall@k, MRR, nDCG
                  คำตอบ: BERTScore, ROUGE-L, LLM-Judge, RAGAS x2
[8] Ablation ---- A dense / B hybrid / C +reranker / D chunk 1024
```

---

## 7. การตัดสินใจเชิงเทคนิค

ทุกข้อต้องอธิบายได้ในการนำเสนอ

| การตัดสินใจ | ค่า | เหตุผล |
|---|---|---|
| ตัวสกัด PDF | PyMuPDF `sort=True` (ถ้าติดตั้งไว้) | เรียง block ตามตำแหน่งบนหน้ากระดาษก่อนดึงข้อความ แก้ปัญหาหมายเหตุริมกระดาษถูกแทรกกลางประโยคตั้งแต่ต้นทาง ต่างจาก pypdf ที่ดึงตามลำดับในไฟล์ |
| Chunking | ตัดตามมาตราก่อน แล้วซอยที่ 512 token / overlap 64 | "มาตรา" คือหน่วยที่สมบูรณ์ทางกฎหมายในตัวเอง การตัดกลางมาตราทำให้เงื่อนไข ("เว้นแต่...") หลุดจากตัวบทหลัก |
| นับ token ด้วย tokenizer จริง | XLM-R SentencePiece | ภาษาไทย 1 คำ ~ 1.8-2.5 subword ถ้านับเป็นคำ chunk จะเกิน `max_seq_length` แล้วถูก truncate โดยไม่มี error |
| หักงบ header | reserve = ความยาว breadcrumb + 12 | ชื่อกฎหมายไทยยาว 30-80 token ถ้าไม่หัก เนื้อหาท้าย chunk หายไปเงียบ ๆ |
| Embedding model | `intfloat/multilingual-e5-base` | เทรนมาเพื่อ retrieval โดยเฉพาะ (asymmetric: `query:` / `passage:`) ต่างจาก paraphrase-mpnet ที่เทรนมาวัดความคล้ายของประโยค |
| Normalize เวกเตอร์ | เปิด | ทำให้ dot product = cosine ถ้าไม่ normalize ระบบจะเอนเอียงไปหา chunk ที่ยาวกว่า |
| Hybrid + RRF (k=60) | BM25 + dense | คำถามกฎหมายมี exact term (เลขมาตรา, "ลาภมิควรได้") ที่ dense จับไม่ติด; RRF ใช้แค่ "อันดับ" จึงไม่ต้อง normalize คะแนนคนละสเกล |
| Cross-encoder rerank | top-20 -> top-5 | bi-encoder เข้ารหัสคำถามกับเอกสารแยกกัน ไม่มี cross-attention; ใช้ของถูกกรองแล้วใช้ของแพงจัดอันดับ |
| ไม่ใช้ vector database | numpy / FAISS Flat | 7,093 เวกเตอร์เล็กเกินกว่าจะต้องใช้ ANN ซึ่งแลกความแม่นยำกับความเร็วที่เราไม่ต้องการ |
| Temperature = 0 | - | RAG ต้องการความคงเส้นคงวาและทำซ้ำได้ ไม่ใช่ความหลากหลาย |
| Prompt บังคับอ้าง [S#] + ปฏิเสธได้ | - | ทำให้วัด faithfulness ได้ และในโดเมนกฎหมาย การเดาผิดอันตรายกว่าการไม่ตอบ |

### Ablation

| Variant | mode | rerank | chunk | ตัวแปรที่เปลี่ยนจากตัวก่อน |
|---|---|---|---|---|
| A | dense | ไม่ | 512/64 | baseline |
| B | hybrid RRF | ไม่ | 512/64 | + สัญญาณ lexical |
| C | hybrid RRF | ใช่ | 512/64 | + การจัดอันดับขั้นที่สอง |
| D | dense | ไม่ | 1024/128 | ขนาด chunk (แยกจากเรื่อง retrieval) |

หลักการ: **เปลี่ยนทีละตัวแปร** ถ้าเปลี่ยนสองอย่างพร้อมกันจะสรุปสาเหตุไม่ได้

Variant D ใช้ chunk คนละขนาด ต้อง ingest แยกก่อน:

```bash
python scripts/ingest.py --config configs/variant_d_chunk1024.yaml
```

---

## 8. โครงสร้างไฟล์

```
src/config.py          ค่าคอนฟิกทั้งหมด (ablation = เปลี่ยน config ไม่ใช่เปลี่ยนโค้ด)
src/env.py             อ่านไฟล์ .env
src/thai_utils.py      normalize + ตัดคำไทย (จุดที่พังง่ายที่สุด)
src/loader.py          โหลดคลังเอกสาร
src/cleaner.py         ทำความสะอาดข้อความจาก PDF
src/chunker.py         chunking 3 ชั้น
src/embedder.py        dense embedding + index
src/sparse.py          BM25
src/retriever.py       RRF + cross-encoder rerank
src/prompts.py         prompt ของระบบและของการประเมิน
src/generator.py       LLM backend (openai / anthropic / hf / echo)
src/pipeline.py        ประกอบทั้งหมด

api/main.py            FastAPI
frontend/src/App.svelte   หน้าเว็บ

eval/metrics.py        เมตริกทั้งหมด
eval/run_eval.py       ประเมิน 1 variant
eval/run_ablation.py   เทียบหลาย variant + ตาราง + กราฟ
eval/failure_analysis.py  จัดกลุ่มความล้มเหลว 5 ประเภท
eval/make_gold_set.py  สร้างร่างชุดทดสอบ

scripts/ingest.py      สร้าง index
scripts/clean_corpus.py  ทำความสะอาดคลัง
scripts/demo.py        เดโมบน terminal
scripts/test_api.py    ทดสอบการเชื่อมต่อ LLM
scripts/check_scanned.py  ตรวจหาหน้าที่เป็นภาพสแกน
scripts/text_quality.py   วัดสระลอย/thai_ratio เป็นตัวเลข
scripts/compare_extractors.py  เทียบตัวสกัด PDF สองตัว
tests/test_offline.py  ทดสอบ logic โดยไม่ต้องโหลดโมเดล
```

---

## สิ่งที่ยังต้องทำด้วยมือ

**ชุดทดสอบ (`data/qa_gold.jsonl`)** — `make_gold_set.py` สร้างได้แค่ร่าง
ทีมต้องอ่านและแก้ทุกข้อ แล้วเปลี่ยน `VERIFIED_BY_HUMAN` เป็นชื่อคนตรวจ

เหตุผลที่ข้ามไม่ได้: ถ้าใช้ LLM สร้างเฉลยแล้วให้ LLM ตัวเดิมมาให้คะแนน
มันจะชอบคำตอบตัวเอง (self-preference bias) ตัวเลข evaluation จะไม่มีความหมาย

องค์ประกอบชุดทดสอบที่แนะนำ อย่างน้อย 40-60 ข้อ:
single-hop 60% / multi-hop 25% / unanswerable 15%
ข้อ unanswerable สำคัญมาก เพราะทดสอบว่าระบบกล้าตอบว่า "ไม่พบข้อมูล" ไหม
และเป็นเนื้อหาของสไลด์ failure analysis

**`AI_AUDIT.md`** — มีโครงให้แล้ว แต่ทุก `[FILL IN: ...]` ต้องแทนที่ด้วยเนื้อหาจริง
รวมถึงต้องบันทึกเซสชันที่ใช้ AI ช่วยสร้าง pipeline นี้ลง Prompt Log ด้วย

---

## ข้อจำกัดที่รู้ตัว

- คำถาม multi-hop ที่ต้องรวมหลายมาตรา ยังพึ่ง top-k ล้วน ๆ ไม่มี query decomposition
- ยังไม่รองรับกฎหมายฉบับแก้ไข ถ้าคลังมีทั้งฉบับเก่าและใหม่ ระบบอาจดึงฉบับที่ถูกยกเลิกแล้ว
- ข้อความจาก PDF ยังมีสระ/วรรณยุกต์หายบางจุด (เช่น "ขอบังคับ") แก้อัตโนมัติไม่ได้เพราะคำที่เหลือยังเป็นคำจริงในพจนานุกรม ต้องเติมใน `data/corrections.txt` เอง
- LLM-as-Judge ใช้โมเดลตระกูลเดียวกับ generator ได้ ทำให้มี self-preference bias
  ทางลด: ใช้คนละค่าย หรือสุ่มตรวจด้วยมนุษย์ 20%
