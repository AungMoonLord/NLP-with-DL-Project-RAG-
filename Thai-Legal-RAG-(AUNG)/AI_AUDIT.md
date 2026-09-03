# AI Audit

**Project:** Domain-Specific Intelligent Q&A System — Thai Legal RAG Pipeline

**Team:**

| # | Name | Student ID |
|---|---|---|
| 1 | ณภัทร วานิชวัตถากร | 67070226 |
| 2 | พลาธิป เหมวุฒิ | 67070255 |
| 3 | อองรักษ์ วณิชชานัย | 67070297 |
| 4 | อื้ออังกูร ชัยวิวัฒน์พร | 67070302 |

**AI Tools Used:** Claude Fable 5, Gemini, Claude Sonnet 5

---
## 1. Tool Inventory

| AI Tool | Primary Use in This Project |
|---|---|
| **Claude Fable 5** | ออกแบบสถาปัตยกรรมระบบ RAG และสร้างโค้ด Baseline Full Pipeline (Ingestion, Chunking, Hybrid Retrieval, Evaluation, Ablation) |
| **Gemini** | ตรวจสอบและแก้ไขบั๊กเชิงลึกระดับ Runtime (Windows MAX_PATH limit, Ingestion crash, Evaluation schema mismatch, Cross-platform scripts) |
| **Claude Sonnet 5** | จัดทำโครงสร้างเอกสารทางเทคนิค บันทึกผลการทดลอง และจัดรูปแบบ Markdown สำหรับ README.md และ AI_AUDIT.md |

---

## 2. Prompt Log

### Entry 1

**Prompt sent:**
> เนื้อคือโปรเจค เกี่ยวกับ RAG ผมอยากให้คุณช่วยเขียนโค้ดสร้าง RAG แบบ Full Pipeline จบเลยได้ไหม ? โดยมีรายละเอียดตามในไฟล์ PDF นี้เลย ส่วน Domain ที่ผมเลือกคือ กฎหมายและตอนนี้ผมเตรียม dataset พร้อมแล้วมีประมาณ 80 Document และเป็นภาษาไทยทั้งหมด

**What AI generated:**
สถาปัตยกรรมระบบ Thai Legal RAG แบบครบวงจร ประกอบด้วย Token-aware chunking, Dense retrieval ด้วยโมเดล SentenceTransformers (`multilingual-e5-base`), BM25 + RRF Hybrid merging, Cross-encoder Reranker (`bge-reranker`), และสคริปต์ประเมินผล BERTScore + LLM-as-Judge

**What you changed or rejected:**
แก้ไขปัญหา Path ในสคริปต์ `src/extract_corpus.py` โดยเพิ่มการคำนวณ Absolute Path อิงจาก Root Directory (`BASE_DIR = Path(__file__).resolve().parent.parent`) และแยกโครงสร้างโฟลเดอร์ให้ชัดเจนระหว่างไฟล์ต้นทางและปลายทาง (`PDF_DIR = "data/pdfs"`, `OUT_DIR = "data/documents"`) เพื่อไม่ให้ Chunker สับสนระหว่างไฟล์ `.pdf` กับ `.txt`

---

### Entry 2

**Prompt sent:**
```
py -c "import json; r=json.load(open('data/extraction_report.json',encoding='utf-8')); [print(x['file'][:50],'->',x.get('error')) for x in r if x.get('method')=='FAILED']"
ประกาศกระทรวงดิจิทัล... -> no such file: 'data/pdfs\ประกาศกระทรวงดิจิทัล...pdf'
ประกาศคณะกรรมการการรักษาความมั่นคงปลอดภัยไซเบอร์... -> no such file...
ทำไมถึงเปิดไฟล์ไม่ได้ทั้งที่ไฟล์มีอยู่จริง?
```

**What AI generated:**
วินิจฉัยว่าติดข้อจำกัดความยาว Path บนระบบปฏิบัติการ Windows (MAX_PATH = 260 characters) เนื่องจากชื่อไฟล์ประกาศกฎหมายมีความยาวเกิน 150 ตัวอักษร และแนะนำให้ใส่ Prefix `\\?\` หน้า Absolute Path ในโค้ด Python

**What you changed or rejected:**
ปฏิเสธแนวทางการใช้ `\\?\` prefix ในโค้ด เพราะพบว่าไลบรารี C/C++ ภายในของ PyMuPDF (`fitz.open()`) ไม่รองรับ Windows Extended Path Prefix ทำให้เกิด Crash และ FileNotFound ต่อเนื่อง จึงเปลี่ยนไปแก้ปัญหาที่ระดับโครงสร้างไฟล์โดยตรง คือ ย้ายโปรเจกต์ไปยัง Root Directory ที่สั้นลง (`D:\thai-legal-rag`) และ Rename ชื่อไฟล์ PDF ภาษาไทยที่ยาวเกินขนาดให้กระชับ เพื่อคงความเข้ากันได้แบบ Cross-platform

---

### Entry 3

**Prompt sent:**
> ช่วยสร้าง testset.json โดยอ้างอิงจากข้อมูลในเอกสารกฎหมายเหล่านี้หน่อย ผมอยากเห็นสัก 20 testset [แนบไฟล์ PDF กฎหมาย 6 ฉบับ]

**What AI generated:**
ชุดข้อมูลทดสอบ 20 ข้อคำถาม-คำตอบ พร้อมระบุชื่อเอกสารอ้างอิงและมาตรากฎหมาย แต่ในตอนท้ายของบล็อก JSON มีการใส่แท็ก Markdown Citation ติดมาด้วย เช่น `[cite: 1, 2, 3, 4, 5, 6]`

**What you changed or rejected:**
ตัดแท็ก cite และ Markdown backticks ส่วนเกินทิ้งทั้งหมด เพื่อรักษาความถูกต้องตามมาตรฐาน JSON Syntax ป้องกันการเกิด `JSONDecodeError` ขณะโหลดเข้าสู่ระบบประเมินผล

---

### Entry 4

**Prompt sent:**
```
Traceback (most recent call last):
  File 'scripts/run_eval.py', line 21, in
    'reference': item['reference_answer'],
KeyError: 'reference_answer'
ทำไม error ล่ะเนี่ย
```

**What AI generated:**
ตรวจพบ Schema Mismatch ระหว่างคีย์ที่สคริปต์ `scripts/run_eval.py` เรียกใช้ (`reference_answer`) กับคีย์ที่อยู่ในไฟล์ `data/testset.json` ซึ่งสร้างไว้ในชื่อ `ground_truth`

**What you changed or rejected:**
ปรับปรุงฟังก์ชันการดึงข้อมูลใน `scripts/run_eval.py` ให้มีความยืดหยุ่น (Robustness) โดยใช้ `ref = item.get("reference_answer") or item.get("ground_truth", "")` แทนการเข้าถึงคีย์แบบเจาะจง เพื่อรองรับชุดข้อมูลทดสอบทั้งสองรูปแบบโดยไม่ต้องแปลงไฟล์ JSON ใหม่ทั้งหมด

---

### Entry 5

**Prompt sent:**
> `scripts/run_ablation.py, line 24 KeyError: 'relevant_docs'` ทำไมถึงรันไม่ผ่าน

**What AI generated:**
ชี้แจงว่าฟังก์ชันคำนวณ Information Retrieval Metrics (`recall_at_k`, `mrr`) ต้องการ Ground Truth ในรูปของ List (`relevant_docs`) แต่ชุดข้อมูลทดสอบเก็บชื่อเอกสารไว้ในคีย์ `source_doc` แบบ String เดี่ยว

**What you changed or rejected:**
แก้ไขลูปประเมินผลใน `scripts/run_ablation.py` ให้ตรวจสอบและดึง `target_docs = item.get("relevant_docs") or [item.get("source_doc")]` พร้อมแปลงเป็น List อัตโนมัติ รวมถึงตรวจสอบ `doc_id` ในก้อน Chunk ที่ดึงขึ้นมาให้รองรับการ Match ชื่อไฟล์ที่มีหรือไม่มีนามสกุล `.pdf`/`.txt`

---

## 3. Decision Journal

| # | Decision | Owner | Reason (in your own words) |
|---|---|---|---|
| 1 | **Corpus Chunking Strategy:** ตัด Chunk แบบ Token-aware ขนาด ~512 tokens พร้อม Overlap | Human | กฎหมายไทยเขียนเป็นรายมาตรา หากตัดสั้นเกินไป (เช่น 256) นิยามและข้อยกเว้นจะขาดออกจากกัน หากตัดยาวเกินไป (เช่น 1024) Vector จะเฉลี่ยความหมายกว้างเกินไปจน Retrieval ดึงมาตราที่เจาะจงไม่เจอ |
| 2 | **Embedding Model:** เลือกใช้ `intfloat/multilingual-e5-base` | AI-suggested, Human-approved | รองรับภาษาไทยได้ดีผ่านการ Pre-train แบบ Cross-lingual สามารถรัน Inference บน CPU ได้อย่างมีประสิทธิภาพตามเงื่อนไขของโปรเจกต์โดยไม่ต้องพึ่ง GPU ขนาดใหญ่ |
| 3 | **Retrieval Architecture:** วางระบบเป็น 2-Stage (Hybrid Search BM25 + Dense ด้วย RRF Merging ตามด้วย Cross-Encoder Reranker) | AI-suggested, Human-approved | เอกสารกฎหมายมีทั้ง Keyword สำคัญเฉพาะเจาะจง (เช่น "มาตรา 4", "ผู้ค้าสินทรัพย์ดิจิทัล") ซึ่ง Dense เพียงอย่างเดียวจับได้ไม่ดีพอ การผสาน BM25 เข้ามาช่วยกู้ Exact Keyword ได้แม่นยำขึ้น |
| 4 | **Evaluation Metric Suite:** ใช้ทั้ง Generation Metrics (BERTScore, LLM-as-Judge) และ Retrieval Metrics (Recall@5, MRR) | Human | เพื่อแยกการประเมิน (Decouple) ประสิทธิภาพของส่วนค้นหาเอกสาร (Retriever) ออกจากส่วนสร้างข้อความ (LLM Generator) ช่วยให้เห็นจุดบกพร่องที่แท้จริงของ Pipeline |
| 5 | **Environment & Text Encoding Policy:** กำหนด `$env:PYTHONUTF8=1` และจัดการ Windows Path Length | Human | Windows PowerShell มี Default Encoding เป็น CP874/CP1252 ซึ่งทำให้การ Print สระภาษาไทยและเปิดไฟล์พัง การบังคับ UTF-8 และปรับโครงสร้าง Path จึงเป็นเงื่อนไขสำคัญต่อความเสถียรของระบบ |

---

## 4. Error Catch Log

### Error 1

**What the AI said:**
เมื่อพบปัญหา Windows เปิดไฟล์ที่มี Path ยาวเกิน 260 ตัวอักษรไม่ได้ AI เสนอให้แก้โค้ดโดยการเติม Extended Path Prefix `\\?\` เข้าไปที่ Path ของไฟล์โดยตรง เช่น `\\?\D:\thai-legal-rag\data\pdfs\...` เพื่อสั่งให้ OS ปลดล็อก MAX_PATH

**Why it was wrong:**
ในทางทฤษฎี Win32 API รองรับ Prefix `\\?\` แต่ในทางปฏิบัติ ฟังก์ชัน `fitz.open()` ของไลบรารี PyMuPDF พัฒนาด้วยแกน C/C++ (MuPDF engine) ซึ่งไม่ได้ส่งค่า Path ผ่าน Windows API ชั้นบนที่แปลง Prefix ดังกล่าว ตัว Engine ภายในจึงมองเห็น `\\?\` เป็นอักขระใน Path จริง ทำให้โปรแกรมโยน Error `FileNotFoundError` ทันที

**How you fixed it:**
ปฏิเสธการแก้โค้ดด้วย Prefix และทำการแก้ไขที่สภาพแวดล้อมจริง โดยย้าย Root Folder ของโปรเจกต์มาไว้ที่ `D:\thai-legal-rag` เพื่อลดความยาว Base Path ลงเกือบ 80 ตัวอักษร และ Rename ไฟล์ PDF ภาษาไทยที่มีชื่อยาวเกินความจำเป็น ทำให้ระบบสามารถเปิดและสกัดข้อความได้ครบ 85/85 ไฟล์ (100%)

---

### Error 2

**What the AI said:**
เมื่อสคริปต์สกัดข้อความ `extract_corpus.py` แจ้งเตือนว่า "⚠️ ไม่เจอคำว่า 'มาตรา' 2 ไฟล์" และแสดง Warning รายการไฟล์สภาฯ ขึ้นมา AI แนะนำให้ปรับ Regex หรืออาจต้องสั่งรัน OCR ใหม่เพราะคาดว่าดึงข้อความไม่สำเร็จ

**Why it was wrong:**
AI ด่วนสรุปว่าการไม่พบคำว่า "มาตรา" คือความล้มเหลวของการทำ Text Extraction แต่ในความเป็นจริงของบริบทกฎหมายไทย ทั้ง 2 ไฟล์ดังกล่าวคือ "ระเบียบสภาผู้แทนราษฎร" ซึ่งรูปแบบทางกฎหมายใช้คำว่า "ข้อ" (เช่น ข้อ 1, ข้อ 2) แทนคำว่า "มาตรา" ตัวข้อความในไฟล์ถูกสกัดออกมาอย่างสมบูรณ์แล้ว ไม่ได้เกิดข้อผิดพลาดในการอ่าน Text แต่อย่างใด

**How you fixed it:**
ตรวจสอบเนื้อหาไฟล์ `.txt` จริงในโฟลเดอร์ `data/documents` เพื่อยืนยันว่าข้อความครบถ้วน จากนั้นปรับเงื่อนไขในตัว Chunker ให้ตรวจจับทั้งคำว่า "มาตรา" สำหรับพระราชบัญญัติ/พระราชกำหนด และคำว่า "ข้อ" สำหรับระเบียบ/ข้อบังคับ เพื่อให้การแบ่ง Chunk ครอบคลุมเอกสารทุกประเภทใน Corpus

---

## 5. Contribution Map

| Team Member | Human Contribution | AI Tools Used |
|---|---|---|
| **[67070297]** | รวบรวมเอกสารกฎหมายไทยเกี่ยวกับ [ภาษี, ธนาคาร, การเงินและการลงทุน] (27 ฉบับ), ตรวจสอบความถูกต้องของ OCR, แก้ไขโครงสร้างไฟล์และ Path บนระบบปฏิบัติการ, ทดลองสร้าง Prototype RAG แบบ Full Pipeline,สร้างไฟล์ README.md, สร้างไฟล์ AI_AUDIT, สร้างไฟล์ json สำหรับ test จำนวน 20 ตัวอย่าง, แก้ Error, ทดลองสร้างเว็บไซต์ (Frontend) และ Deploy RAG (Backend) ขึ้น Web Server, ช่วยเพื่อนในกลุ่มทำสไลด์นำเสนอ | Claude Fable 5, Gemini Flash 3.8, Claude Opus 5, Claude Sonnet 5 |
| **[Member 2]** | "ใส่งานที่ตัวเองทำ" | "ใส่ AI ที่ตัวเองใช้" |
| **[Member 3]** | "ใส่งานที่ตัวเองทำ" | "ใส่ AI ที่ตัวเองใช้" |
| **[Member 4]** | "ใส่งานที่ตัวเองทำ" | "ใส่ AI ที่ตัวเองใช้" |

---

## 6. What Would Break?

> ตอบโดยไม่ใช้ AI

### a) If you removed the reranker (or your second retrieval stage) from your pipeline, what specifically changes in output quality and why?

หากนำ Cross-Encoder Reranker (`bge-reranker`) ออกจากระบบ สิ่งที่จะลดลงทันทีคือ **MRR (Mean Reciprocal Rank)** และความกระชับของ Context ที่ส่งให้ LLM

**กลไกทางเทคนิค:** ตัว Bi-Encoder (Dense Search) และ BM25 (Sparse Search) ทำงานแบบแยกวิเคราะห์ (Independent representation) โดย Bi-Encoder จะบีบอัดทั้งประโยคคำถามและเอกสารออกมาเป็น Vector เดี่ยว ทำให้สูญเสียปฏิสัมพันธ์ระหว่างคำ (Token-to-token cross-attention) ส่งผลให้เอกสารที่มี Keyword ตรงกันแต่อยู่คนละบริบทอาจหลุดขึ้นมาอยู่ใน Top-3 ได้

เมื่อมี Cross-Encoder Reranker ทำหน้าที่ใน Stage 2 ตัว Reranker จะประมวลผลคู่ (Query, Document) พร้อมกันทั้งประโยคผ่าน Full Self-Attention ทำให้โมเดลเข้าใจเงื่อนไข ความสัมพันธ์เชิงลึก และข้อยกเว้นของข้อกฎหมายได้ดีกว่ามาก หากตัดออก Top-1 Document จะมีความแม่นยำน้อยลง ทำให้ LLM ได้รับ Context ที่มี Noise ปน และอาจนำไปสู่การสรุปคำตอบกฎหมายที่ผิดเพี้ยน

### b) Why did you chunk your documents at ~512 tokens? What would happen to retrieval quality if you used 2x or 0.5x that size?

เราเลือกขนาด Chunk ที่ ~512 tokens เพราะสอดคล้องกับขนาดโดยเฉลี่ยของ "หนึ่งมาตราพร้อมวรรคขยายหรือบทยกเว้น" ในโครงสร้างกฎหมายไทย

**หากเพิ่มขนาดเป็น 2x (~1024 tokens):** จะเกิดปัญหา Semantic Dilution (ความหมายเจือจาง) ตัว Embedding Model จะต้องบีบอัดข้อความ 2-3 มาตราที่อาจพูดคนละประเด็นลงในเวกเตอร์ 768 มิติอันเดิม ทำให้ Vector Representation มีค่าเฉลี่ยความหมายที่กว้างเกินไป (Loss of specificity) ส่งผลให้การค้นหาคำถามเจาะจงเฉพาะจุดทำได้แย่ลง อีกทั้งยังเปลือง Context Window ของ LLM โดยมีข้อความที่ไม่เกี่ยวข้องติดไปด้วย

**หากลดขนาดลงเหลือ 0.5x (~256 tokens):** จะเกิดปัญหา Context Fragmentation (บริบทฉีกขาด) ตัวบทกฎหมายไทยมักบัญญัติหลักการไว้ในวรรคหนึ่ง และระบุข้อยกเว้นสำคัญไว้ในวรรคสองหรือวรรคท้าย หากตัดที่ 256 tokens บทยกเว้นจะถูกแยกขาดออกจากหลักการ ทำให้ระบบค้นหาอาจหยิบเฉพาะท่อนหลักการมาตอบโดยไม่เห็นข้อยกเว้น ส่งผลให้ LLM ตอบข้อวินิจฉัยทางกฎหมายผิดพลาดอย่างร้ายแรง

### c) Your ablation compares two system variants. Pick the result that surprised you most and explain the mechanism behind it — not just "variant A scored higher" but why technically.

ผลการทดลองที่น่าสนใจที่สุดคือ: ค่า **Recall@5 เท่ากันที่ 0.5500** ในทุก Variant แต่ค่า **MRR เพิ่มขึ้นอย่างมีนัยสำคัญจาก 0.5167 (Dense) เป็น 0.5500 (Hybrid และ Hybrid+Reranker)**

**กลไกทางเทคนิคเบื้องหลัง:**

*ทำไม Recall@5 ถึงเท่ากัน:* ในชุดทดสอบ 20 ข้อ มีประมาณ 11 ข้อที่เนื้อหาเอกสารเกี่ยวข้องถูกดึงเข้ามาติด 1 ใน 5 อันดับแรกอยู่แล้วตั้งแต่รอบ Dense Search ส่วนอีก 9 ข้อที่เหลือที่ไม่ติด เกิดจากข้อจำกัดของการ Match ชื่อเอกสาร (String Mismatch ระหว่าง `doc_id` กับ `source_doc`) และคำถามบางข้อมีความเป็นนามธรรมสูงมากจนแม้แต่ BM25 ก็ไม่สามารถดึง Keyword เอกสารขึ้นมาติด Top-5 ได้ ทำให้ตัวเลขเพดานการครอบคลุม (Recall Pool) ใน 5 อันดับแรกไม่เปลี่ยนแปลง

*ทำไม MRR จึงขยับสูงขึ้นอย่างชัดเจน:* แม้จำนวนข้อที่ดึงเอกสารถูกต้องจะเท่าเดิม แต่ **ตำแหน่งอันดับ (Rank Position)** ของเอกสารที่ถูกต้องเปลี่ยนไปอย่างมีนัยสำคัญ สำหรับคำถามที่มีศัพท์เทคนิคเฉพาะ เช่น "การประกอบธุรกิจสินทรัพย์ดิจิทัล", "ตั๋วเงินคลัง", "เงินนอกงบประมาณ" ตัว Dense เวกเตอร์เพียงอย่างเดียวมักจัดเอกสารที่ตรงเป้าไปอยู่ในอันดับที่ 3 หรือ 4 แต่เมื่อมีกลไก BM25 เสริมแรงด้วย Reciprocal Rank Fusion (RRF) และตามด้วย Cross-Encoder Reranker ตัวระบบสามารถจับ Term Matching ของศัพท์เฉพาะเหล่านี้ได้ดีขึ้นมาก ส่งผลให้เอกสารที่เกี่ยวข้องถูกดันขึ้นมาอยู่ในอันดับที่ 1 และ 2 ค่าส่วนกลับของอันดับ (1/rank) จึงดีดตัวสูงขึ้นจาก 0.5167 เป็น 0.5500 อย่างชัดเจน ซึ่งส่งผลดีต่อตัว Generator (LLM) โดยตรงที่จะได้อ่านเอกสารสำคัญเป็นลำดับแรก