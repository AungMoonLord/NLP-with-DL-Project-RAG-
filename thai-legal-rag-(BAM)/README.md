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

- อันดับแรกเปิด Terminal แล้วพิมพ์ cd ไปที่โฟลเดอร์ thai-legal-rag-(BAM)

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
python -c "from pathlib import Path; [p.write_text(p.read_text(encoding='utf-8').replace('MODEL_NAME_HERE','gemma-4-E4B-it'),encoding='utf-8') for p in Path('configs').glob('*.yaml')]"
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
python scripts/clean_corpus.py --dry-run        # ดูว่าจะทำความสะอาดอะไร
python scripts/clean_corpus.py --skip-suggest   # ทำความสะอาดจริง (ข้ามขั้นที่ช้า)
python scripts/ingest.py                        # สร้าง chunk + dense + BM25 index
python scripts/ingest.py --skip-dense --skip-bm25   # ทดสอบ chunking อย่างเดียว (เร็ว)

# ใช้งาน
python scripts/demo.py                          # 3 คำถามตัวอย่าง
python scripts/demo.py --interactive --explain  # ถาม-ตอบสด พร้อมดู trace
python -m uvicorn api.main:app --port 8000      # เปิด API

# ประเมินผล
python eval/make_gold_set.py --n 40 --out data/qa_gold.draft.jsonl
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



