# AI Audit

**Project:** Domain-Specific Intelligent Q&A System — Thai Legal RAG Pipeline

**Team:**

| # | Name | Student ID |
|---|---|---|
| 1 | ณภัทร วานิชวัตถากร | 67070226 |
| 2 | พลาธิป เหมวุฒิ | 67070255 |
| 3 | อองรักษ์ วณิชชานัย | 67070297 |
| 4 | เอื้ออังกูร ชัยวิวัฒน์พร | 67070302 |

**AI Tools Used:** Claude Fable 5, Gemini, Claude Sonnet 5, Claude Opus 5

---
## 1. Tool Inventory

| AI Tool | Primary Use in This Project |
|---|---|
| **Claude Fable 5** | ออกแบบสถาปัตยกรรมระบบ RAG และสร้างโค้ด Baseline Full Pipeline (Ingestion, Chunking, Hybrid Retrieval, Evaluation, Ablation) |
| **Gemini Flash 3.8** | ตรวจสอบและแก้ไขบั๊กเชิงลึกระดับ Runtime (Windows MAX_PATH limit, Ingestion crash, Evaluation schema mismatch, Cross-platform scripts) |
| **Claude Sonnet 5** | จัดทำโครงสร้างเอกสารทางเทคนิค บันทึกผลการทดลอง และจัดรูปแบบ Markdown สำหรับ README.md และ AI_AUDIT.md |
| **Claude Opus 5** | ใช้ช่วยตรวจสอบแนวทางเชิงเทคนิคและวิเคราะห์ปัญหาเฉพาะส่วนตามที่สมาชิกทีมใช้งานจริง |
| **Google AI** | ใช้รวบรวมข้อมูลกฏหมาย และคัดกรอกกฎหมายที่จะนำมาใช้งาน |

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
## Entry 6

**Prompt sent:**

ตรวจสอบระบบ RAG ว่า E5 embedding ที่ใช้ใน Dense Retrieval มีการใส่ prefix ถูกต้องทั้งฝั่ง Index และฝั่ง Query หรือไม่ เพราะทราบว่า E5 ต้องใช้ `passage:` ตอนสร้าง embedding ของเอกสาร และ `query:` ตอนค้นหา

**What AI generated:**

AI ตรวจสอบ implementation และชี้ว่าการใช้โมเดลตระกูล E5 ต้องรักษารูปแบบ input ให้สอดคล้องกันทั้งสองฝั่ง โดยเอกสารที่นำไปสร้าง index ควร encode ด้วย `passage: <chunk>` และคำถามของผู้ใช้ควร encode ด้วย `query: <question>` มิฉะนั้น embedding space อาจไม่สอดคล้องกันและทำให้ Retrieval Recall ลดลงโดยระบบยังสามารถทำงานได้โดยไม่เกิด Runtime Error

**What you changed or rejected:**

ตรวจสอบและยืนยัน implementation ใน `indexer.py` และ retrieval code ให้ใช้ prefix ที่ถูกต้องทั้งสองฝั่ง โดยฝั่ง indexing ใช้ `passage:` และฝั่ง query ใช้ `query:` รวมถึงใช้ `normalize_embeddings=True` ทั้งสองฝั่งเพื่อให้ Inner Product ของ FAISS สอดคล้องกับ Cosine Similarity

ผลจากการตรวจสอบพบว่า implementation ปัจจุบันรองรับทั้งสองเงื่อนไขแล้ว จึงไม่ได้เปลี่ยน architecture หลักของระบบ

---

## Entry 7

**Prompt sent:**

ตรวจสอบ Evaluation Pipeline เพราะ testset มีทั้งคำถามที่มีคำตอบอยู่ใน corpus และคำถาม adversarial ที่ตั้งใจถามสิ่งที่ไม่มีใน corpus ถ้านำทุกข้อมาเฉลี่ย BERTScore และ Faithfulness รวมกันจะเกิดปัญหาหรือไม่

**What AI generated:**

AI วิเคราะห์ว่าควรแยกการประเมินออกเป็นสองกลุ่ม เนื่องจากคำถาม adversarial ไม่มี Ground Truth สำหรับนำไปคำนวณ BERTScore แบบเดียวกับคำถามที่ตอบได้

จึงเสนอให้แบ่งเป็น:

* **Answerable**
  * BERTScore
  * Faithfulness
  * Relevance
  * False Abstention Rate
* **Adversarial**
  * Correct Abstention Rate

นอกจากนี้ AI ชี้ว่า Faithfulness ไม่ควรนำคำตอบที่เป็นการปฏิเสธมารวมเฉลี่ยโดยไม่พิจารณา Answer Rate เพราะระบบที่ปฏิเสธทุกคำถามอาจได้คะแนน Faithfulness สูงผิดจริง

**What you changed or rejected:**

ปรับ `run_eval.py` และ logic ใน `src/evaluation.py` ให้แยก Answerable และ Adversarial ออกจากกันอย่างชัดเจน

สำหรับ Answerable มีการรายงาน:

* BERTScore
* Answer Rate
* Faithfulness เฉพาะคำตอบที่ระบบยอมตอบ
* Relevance
* False Abstention Rate

สำหรับ Adversarial รายงาน:

* Correct Abstention Rate

และยังเก็บผลรายคำถามและ Failure Cases เพื่อใช้วิเคราะห์ปัญหาของระบบภายหลัง

---

## Entry 8

**Prompt sent:**

ช่วยตรวจสอบและปรับปรุง `run_ablation.py` สำหรับเปรียบเทียบ Dense, BM25, Hybrid และ Hybrid + Reranker โดยต้องการวัดเฉพาะประสิทธิภาพของ Retrieval ไม่ให้ผลของ LLM Generator เข้ามาปะปน

**What AI generated:**

AI เสนอให้แยก Retrieval Evaluation ออกจาก Full RAG Evaluation และประเมิน 4 Retrieval Variants:

1. `dense`
2. `bm25`
3. `hybrid`
4. `hybrid_rerank`

โดยใช้ Retrieval Metrics ได้แก่:

* Recall@k
* Hit@k
* nDCG@k
* MRR
* Latency

และเพิ่ม Bootstrap 95% Confidence Interval รวมถึง Paired Bootstrap Test สำหรับเปรียบเทียบแต่ละ variant กับ Dense baseline

AI ยังเสนอให้ normalize Document ID ก่อนเปรียบเทียบ เพื่อป้องกันกรณีเดียวกันแต่เขียนเป็น `.pdf`, `.txt`, path เต็ม หรือชื่อไฟล์แตกต่างกัน

**What you changed or rejected:**

ปรับ `run_ablation.py` ให้ใช้เฉพาะคำถาม Answerable ที่มี Gold Document และเรียก Retrieval โดยตรงแทนการเรียก Full Generation Pipeline

เพิ่มการ normalize Document ID ผ่าน `src/metrics.py` และรองรับ Ground Truth หลายรูปแบบ เช่น `relevant_docs`, `gold_docs`, `source_doc` และ `doc_id`

เพิ่มการรายงาน Bootstrap 95% CI, latency และข้อมูล diagnostics ของ reranker เพื่อให้สามารถวิเคราะห์ผล Retrieval ได้โดยไม่ปะปนกับคุณภาพของ LLM Generator

---

## Entry 9

**Prompt sent:**

ตรวจสอบว่าการใช้ LLM ตัวเดียวกันเป็นทั้ง Generator และ LLM-as-Judge มีปัญหาทาง Methodology หรือไม่ เพราะโปรเจกต์ใช้โมเดลเดียวกันเนื่องจากข้อจำกัดด้านค่าใช้จ่าย

**What AI generated:**

AI ระบุว่าการใช้โมเดลเดียวกันสำหรับ Generator และ Judge มีความเสี่ยงด้าน evaluation methodology เนื่องจาก Judge อาจมีแนวโน้มประเมิน output ที่สร้างโดยโมเดลเดียวกันในลักษณะที่ไม่เป็นอิสระจากกัน

AI แนะนำว่าถ้ามีทรัพยากรเพียงพอ ควรใช้ Judge Model ที่แตกต่างจาก Generator Model

**What you changed or rejected:**

ไม่ได้เปลี่ยนโมเดลเนื่องจากข้อจำกัดด้านทรัพยากรและค่าใช้จ่ายของโปรเจกต์

ทีมเลือกเก็บ LLM-as-Judge ไว้ตาม implementation เดิม และบันทึกข้อจำกัดดังกล่าวเป็น **Evaluation Limitation** ของระบบ เพื่อให้ผลการประเมินมีความโปร่งใสและไม่ตีความว่าเป็นการประเมินจากผู้ประเมินที่เป็นอิสระอย่างสมบูรณ์

---

## Entry 10

**Prompt sent:**

ตรวจสอบว่า Evaluation ที่ปรับปรุงแล้วสอดคล้องกับ Requirement ของอาจารย์หรือไม่ โดย Requirement กำหนดให้ใช้ metrics จากรายวิชาอย่างน้อย 2 ตัว เช่น BERTScore, BLEU/ROUGE, LLM-as-Judge, RAGAS Faithfulness และ RAGAS Answer Relevance

**What AI generated:**

AI ตรวจสอบ Evaluation Pipeline และพบว่าระบบมี Metrics ตาม Requirement มากกว่า 2 ตัว ได้แก่:

1. BERTScore
2. LLM-as-Judge — Faithfulness
3. LLM-as-Judge — Relevance

นอกจากนี้ระบบยังมี Retrieval Metrics สำหรับ Ablation เช่น Recall@k, Hit@k, MRR และ nDCG ซึ่งใช้ประเมิน Retrieval โดยเฉพาะ

**What you changed or rejected:**

ไม่เพิ่ม Metric ใหม่ เนื่องจาก Requirement กำหนดเพียงอย่างน้อย 2 Metrics และระบบมี BERTScore และ LLM-as-Judge อยู่แล้ว

ทีมจึงเลือกมุ่งเน้นการทำให้ Evaluation ที่มีอยู่ถูกต้องและแยก Answerable/Adversarial อย่างเหมาะสม แทนการเพิ่ม Metrics ที่ไม่จำเป็น

## 3. Decision Journal

| # | Decision | Owner | Reason (in your own words) |
|---|---|---|---|
| 1 | **Corpus Chunking Strategy:** ตัด Chunk แบบ Token-aware ขนาด ~256 tokens (Overlap 48 tokens) | Human | ขนาด 256 tokens เหมาะกับความยาวเฉลี่ยของอนุมาตราและวรรคสำคัญในกฎหมายไทย ช่วยให้ Vector Representation มีความเฉพาะเจาะจงสูง (High Precision) ไม่เกิดปัญหา Dilution Effect อีกทั้งยังป้องกันไม่ให้ลำดับความยาว (Sequence Length) ล้นเกินขีดจำกัด 512 tokens ของ `multilingual-e5-base` เมื่อรวม prefix "passage: " และลดภาระความหน่วง (Latency) ในการส่ง Context เข้าสู่โมเดล Generator |
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
| **[67070297]** | รวบรวมเอกสารกฎหมายไทยเกี่ยวกับ [ภาษี, ธนาคาร, การเงินและการลงทุน] (27 ฉบับ), ตรวจสอบความถูกต้องของ OCR, แก้ไขโครงสร้างไฟล์และ Path บนระบบปฏิบัติการ, ทดลองสร้าง Prototype RAG แบบ Full Pipeline,สร้างไฟล์ README.md, สร้างไฟล์ AI_AUDIT, สร้างไฟล์ json สำหรับ test จำนวน 20 ตัวอย่าง, แก้ Error | Claude Fable 5, Gemini Flash 3.8, Claude Opus 5, Claude Sonnet 5 |
| **[67070226]** | "รวบรวมเอกสารกฎหมายไทยเกี่ยวกับ [รัฐธรรมนูญ,กฎหมายเลือกตั้ง,รัฐสภา (การเมืองและการปกครอง)], ทดลองสร้าง Prototype RAG แบบ Full Pipeline", ปรับปรุงโค้ดและแก้ไข error | "Claude Sonnet 5, Gemini Flash 3.8" |
| **[67070255]** | "ใส่งานที่ตัวเองทำ" | "ใส่ AI ที่ตัวเองใช้" |
| **[67070302]** | "รวบรวมเอกสารกฎหมายไทยเกี่ยวกับ [กฎหมายแพ่งและพาณิชย], ทำสไลด์นำเสนอ" | "AI ของ Google" |

---

## 6. What Would Break?

> ตอบโดยไม่ใช้ AI

### a) If you removed the reranker (or your second retrieval stage) from your pipeline, what specifically changes in output quality and why?

หากนำ Cross-Encoder Reranker (`bge-reranker`) ออกจากระบบ สิ่งที่จะลดลงทันทีคือ **MRR (Mean Reciprocal Rank)** และความกระชับของ Context ที่ส่งให้ LLM

**กลไกทางเทคนิค:** ตัว Bi-Encoder (Dense Search) และ BM25 (Sparse Search) ทำงานแบบแยกวิเคราะห์ (Independent representation) โดย Bi-Encoder จะบีบอัดทั้งประโยคคำถามและเอกสารออกมาเป็น Vector เดี่ยว ทำให้สูญเสียปฏิสัมพันธ์ระหว่างคำ (Token-to-token cross-attention) ส่งผลให้เอกสารที่มี Keyword ตรงกันแต่อยู่คนละบริบทอาจหลุดขึ้นมาอยู่ใน Top-3 ได้

เมื่อมี Cross-Encoder Reranker ทำหน้าที่ใน Stage 2 ตัว Reranker จะประมวลผลคู่ (Query, Document) พร้อมกันทั้งประโยคผ่าน Full Self-Attention ทำให้โมเดลเข้าใจเงื่อนไข ความสัมพันธ์เชิงลึก และข้อยกเว้นของข้อกฎหมายได้ดีกว่ามาก หากตัดออก Top-1 Document จะมีความแม่นยำน้อยลง ทำให้ LLM ได้รับ Context ที่มี Noise ปน และอาจนำไปสู่การสรุปคำตอบกฎหมายที่ผิดเพี้ยน

### b) Why did you chunk your documents at ~256 tokens? What would happen to retrieval quality if you used 2x or 0.5x that size?

เราเลือกขนาด Chunk ที่ประมาณ 256 tokens พร้อม overlap 48 tokens เพราะเหมาะกับความยาวเฉลี่ยของอนุมาตรา วรรค และเงื่อนไขสำคัญในกฎหมายไทย ทำให้ embedding ของแต่ละ chunk มีความเฉพาะเจาะจงต่อประเด็น (high specificity) มากพอสำหรับการค้นคืน

ขนาดนี้ช่วยลดปัญหา Semantic Dilution ซึ่งเกิดเมื่อ chunk ใหญ่เกินไปและมีหลายประเด็นทางกฎหมายอยู่ใน embedding เดียวกัน ขณะเดียวกัน overlap 48 tokens ช่วยรักษาความต่อเนื่องของข้อความบริเวณรอยตัด โดยเฉพาะกรณีที่เงื่อนไขหลักและข้อยกเว้นอยู่คนละช่วงของข้อความ

หากเพิ่มขนาดเป็น 2 เท่า (~512 tokens): embedding จะต้องสรุปเนื้อหาที่ยาวขึ้นและอาจครอบคลุมหลายมาตราหรือหลายประเด็นในก้อนเดียว ทำให้ความหมายของ vector กว้างขึ้นและลดความสามารถในการค้นหาคำถามที่เจาะจง เช่น คำถามเกี่ยวกับเงื่อนไข ข้อยกเว้น ระยะเวลา หรือบทลงโทษเฉพาะกรณี นอกจากนี้ context ที่ส่งให้ LLM จะมีข้อความที่ไม่เกี่ยวข้องมากขึ้น และเพิ่ม latency ในขั้น generation

หากลดขนาดลงครึ่งหนึ่ง (~128 tokens): จะเสี่ยงต่อปัญหา Context Fragmentation เพราะกฎหมายไทยมักระบุหลักเกณฑ์ในวรรคหนึ่ง แล้วระบุเงื่อนไข ข้อยกเว้น หรือบทลงโทษในวรรคถัดไป หากข้อความถูกแบ่งละเอียดเกินไป ระบบอาจ retrieve ได้เพียงหลักเกณฑ์หลัก แต่ไม่เห็นข้อยกเว้นที่สำคัญ ส่งผลให้คำตอบของ LLM ไม่ครบถ้วนหรือคลาดเคลื่อน

ดังนั้น 256 tokens จึงเป็นจุดสมดุลระหว่างความเฉพาะเจาะจงของ retrieval ความต่อเนื่องของบริบท และต้นทุนในการส่ง context ให้โมเดลสร้างคำตอบ

### c) Your ablation compares two system variants. Pick the result that surprised you most and explain the mechanism behind it — not just "variant A scored higher" but why technically.

ผลการทดลองที่น่าสนใจที่สุดคือ: ค่า **Recall@5 เท่ากันที่ 0.5500** ในทุก Variant แต่ค่า **MRR เพิ่มขึ้นอย่างมีนัยสำคัญจาก 0.5167 (Dense) เป็น 0.5500 (Hybrid และ Hybrid+Reranker)**

**กลไกทางเทคนิคเบื้องหลัง:**

*ทำไม Recall@5 ถึงเท่ากัน:* ในชุดทดสอบ 20 ข้อ มีประมาณ 11 ข้อที่เนื้อหาเอกสารเกี่ยวข้องถูกดึงเข้ามาติด 1 ใน 5 อันดับแรกอยู่แล้วตั้งแต่รอบ Dense Search ส่วนอีก 9 ข้อที่เหลือที่ไม่ติด เกิดจากข้อจำกัดของการ Match ชื่อเอกสาร (String Mismatch ระหว่าง `doc_id` กับ `source_doc`) และคำถามบางข้อมีความเป็นนามธรรมสูงมากจนแม้แต่ BM25 ก็ไม่สามารถดึง Keyword เอกสารขึ้นมาติด Top-5 ได้ ทำให้ตัวเลขเพดานการครอบคลุม (Recall Pool) ใน 5 อันดับแรกไม่เปลี่ยนแปลง

*ทำไม MRR จึงขยับสูงขึ้นอย่างชัดเจน:* แม้จำนวนข้อที่ดึงเอกสารถูกต้องจะเท่าเดิม แต่ **ตำแหน่งอันดับ (Rank Position)** ของเอกสารที่ถูกต้องเปลี่ยนไปอย่างมีนัยสำคัญ สำหรับคำถามที่มีศัพท์เทคนิคเฉพาะ เช่น "การประกอบธุรกิจสินทรัพย์ดิจิทัล", "ตั๋วเงินคลัง", "เงินนอกงบประมาณ" ตัว Dense เวกเตอร์เพียงอย่างเดียวมักจัดเอกสารที่ตรงเป้าไปอยู่ในอันดับที่ 3 หรือ 4 แต่เมื่อมีกลไก BM25 เสริมแรงด้วย Reciprocal Rank Fusion (RRF) และตามด้วย Cross-Encoder Reranker ตัวระบบสามารถจับ Term Matching ของศัพท์เฉพาะเหล่านี้ได้ดีขึ้นมาก ส่งผลให้เอกสารที่เกี่ยวข้องถูกดันขึ้นมาอยู่ในอันดับที่ 1 และ 2 ค่าส่วนกลับของอันดับ (1/rank) จึงดีดตัวสูงขึ้นจาก 0.5167 เป็น 0.5500 อย่างชัดเจน ซึ่งส่งผลดีต่อตัว Generator (LLM) โดยตรงที่จะได้อ่านเอกสารสำคัญเป็นลำดับแรก