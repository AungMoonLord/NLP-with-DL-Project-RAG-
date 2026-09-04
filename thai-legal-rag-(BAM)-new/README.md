# Thai Legal RAG — ระบบถาม-ตอบกฎหมายไทย

ระบบ Q&A เฉพาะโดเมนกฎหมายไทย ทำงานบนคลังเอกสารจริง 85 ฉบับ
(รัฐธรรมนูญ, พระราชบัญญัติประกอบรัฐธรรมนูญ, พระราชบัญญัติ, ข้อบังคับการประชุมรัฐสภา)

ค้นคืนแบบสองขั้น: dense + BM25 รวมด้วย RRF แล้วจัดอันดับซ้ำด้วย cross-encoder
ตอบเป็นภาษาไทยพร้อมอ้างอิงมาตรา/ข้อ ที่ตรวจสอบย้อนกลับได้

**ผลการวัดล่าสุด** (ชุดทดสอบ 36 ข้อ, วัดที่ระดับ chunk)

| Variant | Hit@1 | Hit@3 | Recall@5 | MRR@5 | nDCG@5 | Latency |
|---|---|---|---|---|---|---|
| A: dense อย่างเดียว | 0.486 | 0.800 | 0.804 | 0.660 | 0.669 | 0.88s |
| B: hybrid (dense+BM25+RRF) | 0.457 | 0.800 | 0.833 | 0.645 | 0.669 | 0.80s |
| **C: hybrid + cross-encoder** | **0.743** | **0.971** | **0.852** | **0.857** | **0.801** | 21.5s |

---

## สารบัญ

1. [สิ่งที่ต้องมีก่อน](#1-สิ่งที่ต้องมีก่อน)
2. [ติดตั้ง](#2-ติดตั้ง)
3. [ถ้าแตกไฟล์ zip ทับโฟลเดอร์เดิม](#3-ถ้าแตกไฟล์-zip-ทับโฟลเดอร์เดิม)
4. [เปิดหน้าเว็บ](#4-เปิดหน้าเว็บ)
5. [ประเมินผลและ ablation](#5-ประเมินผลและ-ablation)
6. [คำสั่งทั้งหมด](#6-คำสั่งทั้งหมด)
7. [ปัญหาที่เจอบ่อยและวิธีแก้](#7-ปัญหาที่เจอบ่อยและวิธีแก้)
8. [สถาปัตยกรรมระบบ](#8-สถาปัตยกรรมระบบ)
9. [การตัดสินใจเชิงเทคนิค](#9-การตัดสินใจเชิงเทคนิค)
10. [โครงสร้างไฟล์](#10-โครงสร้างไฟล์)

---

## 1. สิ่งที่ต้องมีก่อน

| สิ่งที่ต้องมี | รายละเอียด |
|---|---|
| Python | 3.10 ขึ้นไป |
| Node.js | 18 ขึ้นไป (เฉพาะถ้าจะใช้หน้าเว็บ) จาก nodejs.org เลือกรุ่น LTS |
| พื้นที่ว่าง | ~6 GB (โมเดล 3.5 GB + index 25 MB + ไลบรารี) |
| RAM | 8 GB ขึ้นไป |
| API key | endpoint ที่รองรับ OpenAI API |

**คลังเอกสารและ index ไม่ได้อยู่ใน repository** (ถูกกันไว้ใน `.gitignore`)
ต้องขอไฟล์ PDF จากทีม แล้ววางใน `data/raw/` เอง

ระบบรันบน CPU ได้ทั้งหมด ไม่ต้องใช้ GPU แต่ขั้นสร้าง index ใช้เวลาประมาณ 30 นาที

---

## 2. ติดตั้ง

### ขั้น 1 — virtual environment และไลบรารี

เปิด Terminal แล้ว `cd` ไปที่โฟลเดอร์โปรเจกต์ก่อน

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

ต้องขึ้น "ผ่านทุกการทดสอบ" ชุดนี้ไม่ใช้โมเดลและไม่ใช้เน็ต
ถ้า ROUGE-L ได้ 0.0 แปลว่า pythainlp ยังไม่ทำงาน ให้ติดตั้งใหม่

### ขั้น 3 — API key

สร้างไฟล์ชื่อ `.env` ที่โฟลเดอร์หลัก (ระดับเดียวกับ README นี้)

```
OPENAI_API_KEY=คีย์ของคุณ
OPENAI_BASE_URL=https://endpoint-ของคุณ/v1
```

> Windows: Notepad จะเติม `.txt` ต่อท้ายเอง ตอนบันทึกต้องเลือก
> "Save as type: All Files" แล้วพิมพ์ชื่อ `.env` ให้ครบ
> **ห้ามเขียน API key ลงในไฟล์ .py และห้าม commit ขึ้น git**

ดูชื่อ model ที่ endpoint รองรับ แล้วใส่ลง config ทั้ง 6 ไฟล์:

```bash
python scripts/test_api.py --list

python -c "from pathlib import Path; [p.write_text(p.read_text(encoding='utf-8').replace('MODEL_NAME_HERE','ชื่อ-model-ของคุณ'),encoding='utf-8') for p in Path('configs').glob('*.yaml')]"

python scripts/test_api.py
```

### ขั้น 4 — วางคลังเอกสารและชุดทดสอบ

- ไฟล์ PDF ทั้งหมด → `data/raw/`
- `testset_BAM.json` → `data/testset_BAM.json`

ตรวจว่าอ่านครบ:

```bash
python scripts/ingest.py --stats-only
python scripts/check_scanned.py       # หาหน้าที่เป็นภาพสแกน (ต้อง OCR)
```

`check_scanned.py` นับตัวอักษรที่ดึงได้ต่อหน้า ถ้าหน้าที่ดึงไม่ออกน้อยกว่า 5%
ไม่ต้องทำ OCR (คลังปัจจุบันอยู่ที่ 0.2% และตรวจด้วยตาแล้วว่าเป็นหน้าปก/หน้าว่าง)

### ขั้น 5 — ทำความสะอาดข้อความ

⚠️ **ขั้นนี้ต้องอ่านจาก `data/raw` เสมอ** ถ้าใช้ config ปกติที่ตั้ง `raw_dir: data/clean`
ไว้แล้ว มันจะไปทำความสะอาดของที่สะอาดอยู่แล้ว (สังเกตได้จาก "ลดลง 0.0%")

สร้าง config เฉพาะสำหรับขั้นนี้:

```bash
python -c "open('configs/_clean.yaml','w',encoding='utf-8').write('raw_dir: data/raw\n')"

python scripts/clean_corpus.py --skip-suggest --config configs/_clean.yaml
```

เปิดอ่าน `data/processed/cleaning_report.md` ก่อนไปต่อ
ถ้าขนาดลดลงเกิน 15% ให้สงสัยว่าลบเนื้อหาจริง แก้ด้วย `--min-run 5` แล้วรันใหม่

จากนั้นบอกระบบให้ใช้ไฟล์สะอาด:

```bash
python -c "from pathlib import Path; [p.write_text('raw_dir: data/clean\n'+p.read_text(encoding='utf-8'),encoding='utf-8') for p in Path('configs').glob('*.yaml') if 'raw_dir' not in p.read_text(encoding='utf-8')]"
```

<details>
<summary>ตัวเลือกเสริม: หาคำที่สระ/วรรณยุกต์หายจาก PDF</summary>

```bash
python scripts/clean_corpus.py --dry-run --config configs/_clean.yaml
```

(ไม่ใส่ `--skip-suggest` — ช้ากว่ามาก) ผลไปที่
`data/processed/corrections_candidates.tsv` เป็นรายการเสนอ เช่น
`กาหนด → กำหนด`, `ขอบังคับ → ข้อบังคับ`

**ระบบจะไม่แก้ให้เอง** ต้องอ่านทุกบรรทัด ลบที่ผิดทิ้ง แล้ว copy ที่เหลือไปต่อท้าย
`data/corrections.txt` เหตุผล: `ขอบังคับ` เป็นคำจริงได้ (ขอ + บังคับ)
พจนานุกรมบอกได้แค่ว่า "คำนี้มีอยู่" ไม่ได้บอกว่า "ในบริบทนี้ควรเป็นคำไหน"

</details>

### ขั้น 6 — สร้าง index

```bash
python scripts/ingest.py
```

**ใช้เวลาประมาณ 30 นาทีบน CPU** ครั้งแรกจะโหลดโมเดล 1.1 GB ด้วย

ตรวจสองบรรทัดนี้ก่อนปล่อยให้รันต่อ:

```
[loader] โหลดสำเร็จ 85 เอกสาร            <- ต้องอ่านจาก data/clean
[chunker] token รวม header: ... เกิน 512: 0 chunk  ✅
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

## ปัญหาที่เจอบ่อยและวิธีแก้

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `pip install` พังตอนอ่าน requirements.txt | pip บน Windows อ่านไฟล์ด้วย locale cp874 ถ้ามีตัวอักษรไทยจะพัง | ไฟล์ปัจจุบันเป็น ASCII ล้วนแล้ว ถ้าแก้เอง อย่าใส่ภาษาไทย |
| `ModuleNotFoundError: openai` | ไม่ได้ติดตั้ง | `pip install openai` |
| `UnicodeEncodeError` ตอนรัน | Windows console ตั้งต้นเป็น cp874 | พิมพ์ `$env:PYTHONUTF8=1` ก่อนรัน |
| `TypeError: unexpected keyword argument 'temperature'` | config ยังเป็น backend anthropic | ตรวจว่า `configs/baseline.yaml` มี `backend: openai` |
| ขึ้น "ยังไม่ได้ตั้งชื่อ model" | ยังไม่ได้แทนที่ `MODEL_NAME_HERE` | ดูขั้น 3 |
| แก้ config แล้วไม่มีอะไรเปลี่ยน | รันโดยไม่ใส่ `--config` และแก้ไฟล์ที่ไม่ใช่ `baseline.yaml` | ระบบใช้ `baseline.yaml` เป็นค่าเริ่มต้น |
| `clean_corpus.py` บอก "ลดลง 0.0%" | `raw_dir` ชี้ไป `data/clean` มันเลยทำความสะอาดของที่สะอาดแล้ว | ใช้ `--config configs/_clean.yaml` |
| `FileNotFoundError: chunks_*.jsonl` | ยังไม่ ingest หรือเปลี่ยน chunk size แล้วยังไม่ ingest ใหม่ | `python scripts/ingest.py` |
| หน้าเว็บขึ้น "เชื่อมต่อ backend ไม่ได้" | uvicorn ไม่ได้รัน หรือใช้ port อื่น | ตรวจ Terminal 1 |
| ตัวเลข chunk บนหน้าเว็บไม่ตรงกับที่เพิ่ง ingest | index เก่าค้างในหน่วยความจำ | ปิด-เปิด uvicorn ใหม่ |
| คำถามแรกช้า 45 วินาที | โหลด cross-encoder 2.3 GB จากดิสก์ | ปกติ คำถามถัดไปจะเร็ว |
| ระบบกลับไปอ่าน `data/raw` เอง | แตก zip ทับ ทำให้ config ถูกเขียนทับ | ดูหัวข้อ 3 |

> ชื่อไฟล์ index ผูกกับ `(embedding model + chunk size + overlap)` เช่น
> `chunks_multilingual-e5-base_c512_o64.jsonl` ดังนั้น variant ที่ใช้ chunk คนละขนาด
> ต้อง ingest แยก และจะไม่เขียนทับกัน



