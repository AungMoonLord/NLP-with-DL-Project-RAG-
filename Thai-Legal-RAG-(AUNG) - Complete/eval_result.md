## Ablation Study

### 1. วัตถุประสงค์

Ablation Study ใช้เพื่อศึกษาว่าองค์ประกอบต่าง ๆ ของระบบ Retrieval มีผลต่อประสิทธิภาพมากน้อยเพียงใด โดยเปรียบเทียบระบบทั้งหมด 4 รูปแบบ:

1. **Dense Retrieval** — ค้นหาด้วย Embedding + FAISS
2. **BM25** — ค้นหาด้วย Keyword / Lexical Matching
3. **Hybrid Retrieval** — รวม Dense Retrieval และ BM25 ด้วย Reciprocal Rank Fusion (RRF)
4. **Hybrid + Reranker** — ใช้ Hybrid Retrieval แล้วนำผลลัพธ์ไปจัดอันดับใหม่ด้วย Cross-Encoder Reranker

---

### 2. ผลการทดลองล่าสุด

| Variant         | Recall@1 | Recall@3 | Recall@5 |    MRR |
| --------------- | -------: | -------: | -------: | -----: |
| Dense           |   0.9000 |   1.0000 |   1.0000 | 0.9417 |
| BM25            |   0.9500 |   0.9500 |   1.0000 | 0.9625 |
| Hybrid          |   0.9500 |   1.0000 |   1.0000 | 0.9750 |
| Hybrid + Rerank |   1.0000 |   1.0000 |   1.0000 | 1.0000 |

> หมายเหตุ: ตัวเลขในส่วนนี้ควรยึดจากผลการทดลองล่าสุดใน `ablation_table.md` และควรใช้ชุดข้อมูลทดสอบเดียวกันในการเปรียบเทียบทุก variant

---

## 3. ความหมายของแต่ละ Metric

### Recall@K

**Recall@K** วัดว่า ในผลลัพธ์ `K` อันดับแรกที่ระบบ Retrieval ส่งกลับมา มีข้อมูลที่ถูกต้อง/Relevant อยู่หรือไม่

พูดง่าย ๆ คือ:

> "ระบบสามารถค้นหาเอกสารหรือ Chunk ที่เราต้องการเจอภายใน Top-K ได้หรือไม่?"

ตัวอย่าง:

ถ้า Ground Truth คือ Chunk ที่เกี่ยวข้องกับคำถาม และระบบคืนผลลัพธ์มา 5 อันดับ:

```text
Rank 1 → ไม่เกี่ยวข้อง
Rank 2 → ไม่เกี่ยวข้อง
Rank 3 → Relevant
Rank 4 → ไม่เกี่ยวข้อง
Rank 5 → ไม่เกี่ยวข้อง
```

กรณีนี้ **Recall@5 = 1**

เพราะ Relevant Chunk ถูกค้นพบภายใน 5 อันดับแรก

### การตีความ

**ยิ่งมากยิ่งดี**

```text
Recall@K = 1.00  → ดีมาก
Recall@K = 0.80  → ค้นพบ Relevant Chunk 80% ของกรณี
Recall@K = 0.50  → ค้นพบเพียงประมาณครึ่งหนึ่ง
```

สำหรับ RAG ค่า Recall สำคัญ เพราะถ้า Retrieval หา Chunk ที่เกี่ยวข้องไม่เจอเลย ต่อให้ LLM เก่งแค่ไหน ก็ไม่มีข้อมูลที่ถูกต้องให้ใช้ตอบ

---

## 4. Recall@1

Recall@1 ดูเฉพาะผลลัพธ์อันดับ 1

คำถามคือ:

> "Chunk ที่ Relevant สามารถถูกค้นพบเป็นอันดับแรกหรือไม่?"

ผลการทดลอง:

| Variant         |   Recall@1 |
| --------------- | ---------: |
| Dense           |     0.9000 |
| BM25            |     0.9500 |
| Hybrid          |     0.9500 |
| Hybrid + Rerank | **1.0000** |

ดังนั้นในชุดทดสอบนี้ Hybrid + Rerank สามารถนำ Relevant Result ขึ้นมาอยู่ในอันดับแรกได้ครบทุกกรณี

**Metric นี้ยิ่งมากยิ่งดี**

---

## 5. Recall@3

Recall@3 ดูว่า Relevant Result ถูกค้นพบภายใน Top 3 หรือไม่

ผลการทดลอง:

| Variant         | Recall@3 |
| --------------- | -------: |
| Dense           |   1.0000 |
| BM25            |   0.9500 |
| Hybrid          |   1.0000 |
| Hybrid + Rerank |   1.0000 |

Dense, Hybrid และ Hybrid + Rerank มีค่า 1.0000 ในการทดลองนี้

หมายความว่า Relevant Result ถูกค้นพบภายใน Top 3 ในทุกกรณีของชุดทดสอบนั้น

**Metric นี้ยิ่งมากยิ่งดี**

---

## 6. Recall@5

Recall@5 ดูว่า Relevant Result ถูกค้นพบภายใน Top 5 หรือไม่

ผลการทดลอง:

| Variant         | Recall@5 |
| --------------- | -------: |
| Dense           |   1.0000 |
| BM25            |   1.0000 |
| Hybrid          |   1.0000 |
| Hybrid + Rerank |   1.0000 |

ทุก variant ได้ค่า 1.0000

สิ่งที่ควรระวังคือ **ไม่ได้หมายความว่าทุก variant มีประสิทธิภาพเท่ากันทั้งหมด**

เพราะ Recall@5 สนใจเพียงว่า Relevant Result "อยู่ใน Top 5 หรือไม่" แต่ไม่ได้สนใจว่าอยู่ลำดับที่ 1 หรือ 5

ดังนั้นแม้ทุกระบบจะมี Recall@5 = 1.0 แต่ลำดับของ Relevant Result อาจแตกต่างกันได้ ซึ่งเป็นเหตุผลที่ต้องดู **MRR** ร่วมด้วย

**Metric นี้ยิ่งมากยิ่งดี**

---

# 7. MRR (Mean Reciprocal Rank)

MRR = Mean Reciprocal Rank

Metric นี้วัดว่า **Relevant Result ถูกจัดไว้ในอันดับที่เท่าไร**

หลักการคือ:

```text
อันดับ 1 → 1/1 = 1.00
อันดับ 2 → 1/2 = 0.50
อันดับ 3 → 1/3 = 0.33
อันดับ 4 → 1/4 = 0.25
อันดับ 5 → 1/5 = 0.20
```

จากนั้นนำค่าของแต่ละ Query มาหาค่าเฉลี่ย

ดังนั้น:

> **ยิ่ง Relevant Result อยู่ด้านบนมากเท่าไร MRR ก็ยิ่งสูง**

ผลการทดลอง:

| Variant         |        MRR |
| --------------- | ---------: |
| Dense           |     0.9417 |
| BM25            |     0.9625 |
| Hybrid          |     0.9750 |
| Hybrid + Rerank | **1.0000** |

ผลนี้แสดงให้เห็นว่า ในชุดทดสอบนี้ Hybrid + Rerank สามารถจัด Relevant Result ให้อยู่ในอันดับที่ดีขึ้นจนได้ MRR = 1.0000

**Metric นี้ยิ่งมากยิ่งดี**

---

# 8. สรุปความหมายของ Metrics

| Metric   | วัดอะไร                                          | ค่าที่ดี | ค่าที่ไม่ดี |
| -------- | ------------------------------------------------ | -------- | ----------- |
| Recall@1 | Relevant Result อยู่ในอันดับ 1 หรือไม่           | ใกล้ 1   | ใกล้ 0      |
| Recall@3 | Relevant Result อยู่ใน Top 3 หรือไม่             | ใกล้ 1   | ใกล้ 0      |
| Recall@5 | Relevant Result อยู่ใน Top 5 หรือไม่             | ใกล้ 1   | ใกล้ 0      |
| MRR      | Relevant Result ถูกจัดอยู่ใกล้อันดับต้น ๆ แค่ไหน | ใกล้ 1   | ใกล้ 0      |

**ทุก Metric ในตารางนี้เป็น Metric ที่ยิ่งมากยิ่งดี**

---

# 9. การตีความ Ablation

ผลการทดลองสามารถอธิบายตามกลไกของแต่ละระบบได้ดังนี้

### Dense Retrieval

Dense Retrieval ใช้ Embedding เพื่อจับความสัมพันธ์เชิงความหมายระหว่าง Query และ Chunk

ข้อดีคือสามารถค้นหาข้อความที่มีความหมายใกล้เคียงกัน แม้ไม่ได้ใช้คำเหมือนกันทั้งหมด

ผล:

```text
Recall@1 = 0.9000
Recall@3 = 1.0000
Recall@5 = 1.0000
MRR      = 0.9417
```

แสดงว่า Dense Retrieval สามารถค้นหา Relevant Chunk ได้ดี แต่ยังมีบางกรณีที่ Relevant Result ไม่ได้ถูกจัดขึ้นเป็นอันดับ 1

---

### BM25

BM25 เป็น Lexical Retrieval ที่เน้นการจับคู่คำใน Query กับคำในเอกสาร

เหมาะกับ Query ที่มีคำเฉพาะ เช่น:

* ชื่อกฎหมาย
* เลขมาตรา
* คำศัพท์ทางกฎหมาย
* ชื่อเฉพาะ

ผล:

```text
Recall@1 = 0.9500
Recall@3 = 0.9500
Recall@5 = 1.0000
MRR      = 0.9625
```

ผลลัพธ์แสดงว่า BM25 สามารถค้นหา Relevant Result ได้ดีในชุดทดสอบนี้ โดยเฉพาะกรณีที่มีคำสำคัญตรงกับเอกสาร

---

### Hybrid Retrieval

Hybrid รวม Dense Retrieval และ BM25 เพื่อใช้ประโยชน์จากทั้ง:

```text
Dense
→ Semantic Matching

BM25
→ Lexical / Keyword Matching
```

จากนั้นใช้ RRF เพื่อรวม Ranking จากทั้งสองวิธี

ผล:

```text
Recall@1 = 0.9500
Recall@3 = 1.0000
Recall@5 = 1.0000
MRR      = 0.9750
```

MRR เพิ่มจาก:

```text
Dense  = 0.9417
BM25   = 0.9625
Hybrid = 0.9750
```

ผลนี้สอดคล้องกับแนวคิดของ Hybrid Retrieval ที่พยายามรวมข้อดีของ Semantic Search และ Lexical Search

---

### Hybrid + Reranker

ขั้นตอนนี้นำ Candidate ที่ได้จาก Hybrid Retrieval มาจัดอันดับใหม่ด้วย Cross-Encoder Reranker

ต่างจาก Bi-Encoder Embedding ที่แปลง Query และ Document แยกกันเป็น Vector แล้วเปรียบเทียบกัน Cross-Encoder สามารถพิจารณา Query และ Candidate Chunk ร่วมกันเพื่อประเมินความเกี่ยวข้อง

ผล:

```text
Recall@1 = 1.0000
Recall@3 = 1.0000
Recall@5 = 1.0000
MRR      = 1.0000
```

ผลนี้แสดงว่าในการทดลองชุดนี้ Reranker สามารถจัด Relevant Result ให้อยู่ในอันดับแรกได้ทุกกรณี

อย่างไรก็ตาม ไม่ควรสรุปจากผลนี้เพียงอย่างเดียวว่า Reranker จะให้ผลดีในทุก Dataset เนื่องจากผลการทดลองขึ้นอยู่กับขนาดและลักษณะของ Evaluation Dataset ด้วย

---

# 10. สรุป Ablation แบบสั้น

ผลการทดลองล่าสุด:

```text
Dense
    ↓
Recall@1 = 0.9000
MRR      = 0.9417

BM25
    ↓
Recall@1 = 0.9500
MRR      = 0.9625

Hybrid
    ↓
Recall@1 = 0.9500
MRR      = 0.9750

Hybrid + Rerank
    ↓
Recall@1 = 1.0000
MRR      = 1.0000
```

ผลลัพธ์แสดงให้เห็นว่าแต่ละขั้นตอนเพิ่มความสามารถในการจัดอันดับ Relevant Result ในชุดทดสอบนี้ โดยเฉพาะเมื่อเพิ่ม Hybrid Retrieval และ Reranking

อย่างไรก็ตาม การตีความควรจำกัดอยู่ภายใน Evaluation Dataset ที่ใช้ทดลอง และไม่ควรสรุปว่าเป็นผลที่จะเกิดขึ้นกับข้อมูลทุกประเภท
