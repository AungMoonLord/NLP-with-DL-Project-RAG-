"""Central configuration — ทุกค่าที่ต้อง 'defend' ตอน Q&A อยู่ที่นี่"""

# ---------- Chunking ----------
# เหตุผล: เอกสารกฎหมายไทยมีโครงสร้างเป็น "มาตรา" สั้นๆ
# chunk เล็กเกินไป (128) จะตัดเนื้อความมาตราขาด, ใหญ่เกินไป (1024)
# จะรวมหลายมาตราจน embedding เจือจาง (semantic dilution)
# → 256 tokens + overlap 48 คือจุดสมดุล (ควรทดลอง 128/256/512 แล้วรายงานใน ablation)
CHUNK_SIZE = 256          # หน่วย: token ของ tokenizer โมเดล embedding
CHUNK_OVERLAP = 48

# ---------- Embedding ----------
# multilingual-e5-base: รองรับภาษาไทยดี, รันบน CPU ได้, ต้องใส่ prefix query:/passage:
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ---------- Retrieval ----------
TOP_K_RETRIEVE = 20       # จำนวน candidate ก่อน rerank/merge
TOP_K_FINAL = 5           # จำนวน context ที่ส่งเข้า LLM
RRF_K = 60                # ค่ามาตรฐานจาก paper RRF (Cormack et al., 2009)

# ---------- Reranker ----------
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder multilingual

# ---------- LLM (Generation + LLM-as-Judge) ----------
# ชี้ไปที่ OpenAI-compatible endpoint ใดก็ได้ (OpenAI / Ollama / vLLM / etc.)
# LLM_BASE_URL = "https://api.openai.com/v1"
# LLM_API_KEY = "YOUR_API_KEY"
# LLM_MODEL = "gpt-4o-mini"
# JUDGE_MODEL = "gpt-4o-mini"


# ---------- LLM (Generation + LLM-as-Judge) ----------
# ชี้ไปที่ OpenAI-compatible endpoint ของคลาส
LLM_BASE_URL = "https://llm.nat-d.uk/v1"
LLM_API_KEY = ""
LLM_MODEL = "gemma-4-E4B-it"
JUDGE_MODEL = "gemma-4-E4B-it"

# ---------- Paths ----------
DOCS_DIR = "data/documents"
INDEX_DIR = "artifacts"
TESTSET_PATH = "data/testset.json"
RESULTS_PATH = "artifacts/eval_results.json"

