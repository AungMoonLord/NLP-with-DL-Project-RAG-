# AI Audit — Thai Legal RAG: ระบบถาม-ตอบกฎหมายไทย

**Team:** [FILL IN: ชื่อสมาชิกทั้ง 3-4 คน]
**AI Tools Used:** [FILL IN: ระบุทุกตัวที่ใช้จริง เช่น Claude (chat), Claude Code, GitHub Copilot, ChatGPT]

> ⚠️ ไฟล์นี้เป็น **โครง** ที่ยังไม่สมบูรณ์ ทุก `[FILL IN: ...]` ต้องถูกแทนที่ด้วยเนื้อหาจริง
> โจทย์ระบุชัดว่า "any unfilled section scores zero"
> **สำคัญ:** เซสชันที่ใช้ AI ช่วยสร้าง pipeline นี้ ต้องถูกบันทึกใน Prompt Log ด้วย
> การไม่บันทึกคือการรายงานเท็จ และเป็นสิ่งที่ checker จับได้ง่ายที่สุด

---

## 1. Tool Inventory

| AI Tool | Primary Use in This Project |
|---|---|
| [FILL IN: เช่น Claude] | [FILL IN: เช่น สร้างโครง pipeline ทั้งหมด (chunker, retriever, eval harness) จาก PDF โจทย์] |
| [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] |

---

## 2. Prompt Log (ขั้นต่ำ 5 entries)

### Entry 1 — สร้างโครง pipeline
**Prompt sent:**
> [FILL IN: วางข้อความจริงที่ส่งไป]

**What AI generated:**
[FILL IN: สรุปสิ่งที่ AI สร้าง เช่น โครงสร้างโปรเจกต์ 12 ไฟล์ ประกอบด้วย chunker แบบ 3 ชั้น, RRF fusion, evaluation harness 5 เมตริก]

**What you changed or rejected:**
[FILL IN: ต้องเจาะจง เช่น "เปลี่ยน embedding model จาก X เป็น Y เพราะ...", "ตัดฟีเจอร์ Z ทิ้งเพราะเกินขอบเขต", "แก้ค่า overlap จาก 64 เป็น 32 หลังทดสอบกับคลังจริง"]

---

### Entry 2 — [FILL IN: หัวข้อ เช่น การเลือก embedding model สำหรับภาษาไทย]
**Prompt sent:**
> [FILL IN]

**What AI generated:**
[FILL IN]

**What you changed or rejected:**
[FILL IN]

---

### Entry 3 — [FILL IN: เช่น การออกแบบ chunking สำหรับเอกสารกฎหมาย]
**Prompt sent:**
> [FILL IN]

**What AI generated:**
[FILL IN]

**What you changed or rejected:**
[FILL IN]

---

### Entry 4 — [FILL IN: เช่น การ implement RAGAS faithfulness ด้วยตัวเอง]
**Prompt sent:**
> [FILL IN]

**What AI generated:**
[FILL IN]

**What you changed or rejected:**
[FILL IN]

---

### Entry 5 — [FILL IN: เช่น การ debug ปัญหาที่พบตอนรันจริง]
**Prompt sent:**
> [FILL IN]

**What AI generated:**
[FILL IN]

**What you changed or rejected:**
[FILL IN]

---

## 3. Decision Journal (ขั้นต่ำ 5 decisions)

| # | Decision | Owner | Reason (เขียนด้วยคำของตัวเอง) |
|---|---|---|---|
| 1 | Chunk size = 512 tokens, overlap 64, ตัดตามขอบเขต "มาตรา" ก่อน | [FILL IN: Human / AI-suggested, Human-approved / AI-decided] | [FILL IN: ต้องอ้างผลทดลองจริงของทีม เช่น ทดสอบ 256/512/1024 แล้ว recall@5 ได้เท่าไรบ้าง] |
| 2 | Embedding model = `intfloat/multilingual-e5-base` | [FILL IN] | [FILL IN: ทำไมไม่ใช้ paraphrase-multilingual-mpnet หรือ bge-m3 — ต้องมีเหตุผลเชิงประจักษ์หรือข้อจำกัดด้านทรัพยากร] |
| 3 | Retrieval = hybrid (dense + BM25) รวมด้วย RRF k=60 | [FILL IN] | [FILL IN] |
| 4 | Cross-encoder reranking top-20 → top-5 | [FILL IN] | [FILL IN] |
| 5 | เมตริกที่ใช้ = [FILL IN: ระบุ 2+ เมตริกที่ทีมใช้จริง] | [FILL IN] | [FILL IN: แต่ละเมตริกวัดคนละอย่างอย่างไร ทำไมต้องใช้มากกว่าหนึ่ง] |
| 6 | Ablation = A/B/C/D เปลี่ยนทีละตัวแปร | [FILL IN] | [FILL IN] |

---

## 4. Error Catch Log (ขั้นต่ำ 2 entries)

> ⚠️ ห้ามแต่งขึ้น ต้องเป็นข้อผิดพลาดที่ทีมเจอจริง
> (ระหว่างพัฒนา pipeline นี้พบข้อผิดพลาดจริงอย่างน้อย 1 ข้อ ดูหมายเหตุด้านล่าง —
> แต่ทีมต้องยืนยันด้วยตัวเองว่าเจอจริงในสภาพแวดล้อมของทีม ก่อนนำมาเขียน)

### Error 1
**What the AI said:**
[FILL IN]

**Why it was wrong:**
[FILL IN: อ้างอิงคอนเซ็ปต์ในวิชา เช่น "ใช้ dot product โดยไม่ normalize เวกเตอร์ ซึ่งไม่เท่ากับ cosine similarity"]

**How you fixed it:**
[FILL IN]

---

### Error 2
**What the AI said:**
[FILL IN]

**Why it was wrong:**
[FILL IN]

**How you fixed it:**
[FILL IN]

---

<!--
หมายเหตุสำหรับทีม (ลบบล็อกนี้ก่อนส่ง):
ข้อผิดพลาดที่ถูกพบและแก้ระหว่างพัฒนา — ตรวจสอบซ้ำเองก่อนนำไปเขียนเป็น Error 1/2
(A) โค้ดชุดแรกใช้ unicodedata.normalize("NFKC") กับข้อความไทย ซึ่งทำให้สระอำ (U+0E33)
    ถูกแตกเป็น นิคหิต (U+0E4D) + สระอา (U+0E32) คำว่า "ชำระ" จึงไม่ตรงพจนานุกรมของ
    newmm ทำให้การตัดคำผิด และ BM25 หา term ไม่เจอ โดยไม่มี error แจ้งเตือน
    วิธีตรวจสอบซ้ำ:  python -c "import unicodedata;print([hex(ord(c)) for c in unicodedata.normalize('NFKC','ชำระ')])"
    วิธีแก้ที่ใช้: เพิ่มฟังก์ชัน recompose_sara_am() ใน src/thai_utils.py
(B) fallback tokenizer ตอนแรกใช้การ split ด้วยช่องว่าง ซึ่งกับภาษาไทยได้ 1 token ทั้งประโยค
    ทำให้ ROUGE-L คืนค่า 0.0 เสมอ (ดู tests/test_offline.py ที่จับเคสนี้ไว้)
-->

---

## 5. Contribution Map

| Team Member | Human Contribution | AI Tools Used |
|---|---|---|
| [FILL IN: ชื่อ] | [FILL IN: เช่น รวบรวมและทำความสะอาดคลังเอกสาร 80 ฉบับ, ออกแบบกลยุทธ์ chunking] | [FILL IN] |
| [FILL IN] | [FILL IN: เช่น implement BM25 + RRF, จูน k] | [FILL IN] |
| [FILL IN] | [FILL IN: เช่น สร้าง gold set 50 ข้อและตรวจด้วยมือ, evaluation harness] | [FILL IN] |
| [FILL IN] | [FILL IN: เช่น ablation, failure analysis, สไลด์] | [FILL IN] |

---

## 6. What Would Break? (ตอบโดยไม่ใช้ AI)

**a) ถ้าถอด cross-encoder reranker (ขั้นค้นคืนขั้นที่สอง) ออกจาก pipeline คุณภาพผลลัพธ์เปลี่ยนอย่างไร และเพราะอะไร**

[FILL IN: ตอบด้วยตัวเลขจาก ablation ของทีมเอง — เทียบ variant B กับ C
แนวทางที่ควรพูดถึง: reranker ไม่ได้เพิ่ม recall (มันเลือกจาก candidate ชุดเดิม)
แต่เพิ่ม precision ที่ตำแหน่งต้น ๆ ดังนั้น MRR/nDCG ควรขยับ ส่วน recall@20 ไม่ควรเปลี่ยน
และให้เชื่อมกับกลไก: cross-encoder เห็น query กับ chunk พร้อมกันจึงจับความสัมพันธ์ระดับคำได้]

**b) ทำไมถึงเลือก chunk ที่ [FILL IN: ระบุขนาดจริงที่ใช้] token และถ้าใช้ 2 เท่า หรือ ครึ่งหนึ่ง จะเกิดอะไรขึ้น**

[FILL IN: ตอบจากผลจริงของ variant D (1024) และ E (256) ที่ทีมรัน
กรอบการตอบ: chunk เล็ก → เวกเตอร์เจาะจงขึ้น precision ดี แต่เสี่ยงตัดเงื่อนไขในมาตราขาดกลางคัน
chunk ใหญ่ → บริบทครบ แต่เวกเตอร์เดียวต้องแทนหลายประเด็น สัญญาณเจือจาง (topic dilution)
และเปลืองงบ token ตอนส่งเข้า LLM]

**c) เลือกผลจาก ablation ที่ "เซอร์ไพรส์" ที่สุด แล้วอธิบายกลไกเบื้องหลัง**

[FILL IN: ต้องเป็นผลจริงของทีม ห้ามเขียนแค่ "variant C ชนะ"
ต้องอธิบายว่าทำไมในเชิงกลไก เช่น ดูจาก failure_*.md ว่าคำถามประเภทไหนที่ B ชนะ A
แล้วเชื่อมกับลักษณะของคำถามนั้น]
