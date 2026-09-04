# Ablation Results — Thai Legal RAG

| Metric | A: Dense-only | B: Hybrid (dense+BM25+RRF) | C: Hybrid + CrossEncoder |
|---|---|---|---|
| Recall@5 | 0.8041 | 0.8333 | 0.8517 |
| MRR@5 | 0.6605 | 0.6452 | 0.8571 |
| nDCG@5 | 0.6692 | 0.6687 | 0.8014 |
| Latency (s) | 0.8772 | 0.7995 | 21.4832 |

## Variants

- **A: Dense-only** — mode=`dense`, rerank=`False`, chunk=`512/64`, top_k=`5`
- **B: Hybrid (dense+BM25+RRF)** — mode=`hybrid`, rerank=`False`, chunk=`512/64`, top_k=`5`
- **C: Hybrid + CrossEncoder** — mode=`hybrid`, rerank=`True`, chunk=`512/64`, top_k=`5`

## การตีความ (เขียนเองหลังเห็นตัวเลขจริง)

> อย่าเขียนแค่ 'variant C ชนะ' — ต้องอธิบายกลไก เช่น
> คำถามที่มีเลขมาตราชัดเจนได้ประโยชน์จาก BM25 มากที่สุด เพราะ ...
> ส่วนคำถามเชิงนิยาม dense ทำได้ดีอยู่แล้ว เพราะ ...