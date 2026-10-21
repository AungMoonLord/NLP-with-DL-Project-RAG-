# """Central configuration — ทุกค่าที่ต้อง 'defend' ตอน Q&A อยู่ที่นี่"""

# # ---------- Chunking ----------
# # เหตุผล: เอกสารกฎหมายไทยมีโครงสร้างเป็น "มาตรา" สั้นๆ
# # chunk เล็กเกินไป (128) จะตัดเนื้อความมาตราขาด, ใหญ่เกินไป (1024)
# # จะรวมหลายมาตราจน embedding เจือจาง (semantic dilution)
# # → 256 tokens + overlap 48 คือจุดสมดุล (ควรทดลอง 128/256/512 แล้วรายงานใน ablation)
# CHUNK_SIZE = 256          # หน่วย: token ของ tokenizer โมเดล embedding
# CHUNK_OVERLAP = 48

# # ---------- Embedding ----------
# # multilingual-e5-base: รองรับภาษาไทยดี, รันบน CPU ได้, ต้องใส่ prefix query:/passage:
# EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# # ---------- Retrieval ----------
# TOP_K_RETRIEVE = 20       # จำนวน candidate ก่อน rerank/merge
# TOP_K_FINAL = 5           # จำนวน context ที่ส่งเข้า LLM
# RRF_K = 60                # ค่ามาตรฐานจาก paper RRF (Cormack et al., 2009)

# # ---------- Reranker ----------
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder multilingual

# # ---------- LLM (Generation + LLM-as-Judge) ----------
# # ชี้ไปที่ OpenAI-compatible endpoint ใดก็ได้ (OpenAI / Ollama / vLLM / etc.)
# # LLM_BASE_URL = "https://api.openai.com/v1"
# # LLM_API_KEY = "sk-qLAwKH1pgFQIxG-2fRCGpg"
# # LLM_MODEL = "gpt-4o-mini"
# # JUDGE_MODEL = "gpt-4o-mini"


# # ---------- LLM (Generation + LLM-as-Judge) ----------
# # ชี้ไปที่ OpenAI-compatible endpoint ของคลาส
# LLM_BASE_URL = "https://llm.nat-d.uk/v1"
# LLM_API_KEY = ""
# LLM_MODEL = "gemma-4-E4B-it"
# JUDGE_MODEL = "gemma-4-E4B-it"

# # ---------- Paths ----------
# DOCS_DIR = "data/documents"
# INDEX_DIR = "artifacts"
# TESTSET_PATH = "data/testset.json"
# RESULTS_PATH = "artifacts/eval_results.json"

"""Central configuration — ทุกค่าที่ต้อง 'defend' ตอน Q&A อยู่ที่นี่"""
import os
from pathlib import Path

# ---------- Base Paths (Cross-Platform) ----------
ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "data" / "documents"
INDEX_DIR = ROOT_DIR / "artifacts"
TESTSET_PATH = ROOT_DIR / "data" / "testset.json"
RESULTS_PATH = ROOT_DIR / "artifacts" / "eval_results.json"

# ---------- Chunking ----------
# 256 tokens + overlap 48 สอดคล้องกับขนาดเฉลี่ย 1 มาตรากฎหมายไทย
CHUNK_SIZE = 256          # หน่วย: token ของ tokenizer
CHUNK_OVERLAP = 48

# ---------- Embedding ----------
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ---------- Retrieval & Reranker ----------
TOP_K_RETRIEVE = 30       # ดึง candidate จาก Dense / BM25 ก่อนนำมารวมและ Rerank (ปรับขึ้นจาก 20 เพื่อเพิ่ม Recall pool)
TOP_K_FINAL = 5           # จำนวน Context สูงสุดที่ส่งเข้า LLM
RRF_K = 60                # Smoothing constant มาตรฐาน RRF (Cormack et al., 2009)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------- Evaluation Metrics Cutoffs ----------
EVAL_K_VALUES = [1, 3, 5] # สำหรับดูความแม่นยำในระดับ Rank ต่างๆ

# ---------- LLM & Judge (Class Endpoint) ----------
LLM_BASE_URL = "https://llm.nat-d.uk/v1"
# ดึงจาก Environment Variable ก่อน ถ้าไม่มีให้ fallback ไปที่คีย์เดิม
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-qLAwKH1pgFQIxG-2fRCGpg")

# ใช้โมเดล gemma-4-E4B-it ทั้งคู่ตาม Resource ที่ได้รับจากคลาส
LLM_MODEL = "gemma-4-E4B-it"
JUDGE_MODEL = "gemma-4-E4B-it"
JUDGE_TEMPERATURE = 0.0

# ข้อความมาตรฐานเมื่อระบบปฏิเสธการตอบเพื่อลด Hallucination
ABSTAIN_TEXT = "ไม่พบข้อมูลในเอกสารที่มี"

