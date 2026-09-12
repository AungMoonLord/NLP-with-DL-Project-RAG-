"""Central configuration for the Thai Legal RAG system.

ทุกค่าที่เป็น "design decision" ถูกดึงออกมาไว้ที่นี่ที่เดียว
เพื่อให้การทำ ablation study = เปลี่ยน config ไม่ใช่เปลี่ยนโค้ด
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ChunkConfig:
    # ขนาด chunk วัดด้วย tokenizer ของ "embedding model" ไม่ใช่จำนวนคำ
    max_tokens: int = 512
    min_tokens: int = 80          # chunk ที่สั้นกว่านี้จะถูก merge กับมาตราถัดไป
    overlap_tokens: int = 64      # ~12.5% ของ max_tokens
    respect_law_structure: bool = True   # ตัดตามขอบเขต "มาตรา"/"ข้อ" ก่อนเสมอ
    merge_short_sections: bool = True
    prepend_breadcrumb: bool = True      # ใส่ ชื่อกฎหมาย > ลักษณะ > หมวด นำหน้าตอน embed
    # กันข้อความถูก truncate เงียบ ๆ: หัก token ของ header ออกจากงบก่อนซอย
    reserve_header_tokens: bool = True
    # chunk ที่สั้นกว่านี้และ merge ไม่ได้ ให้ทิ้ง (เศษจากการแปลง PDF เช่น เลขหน้าเดี่ยว ๆ)
    drop_below_tokens: int = 15
    strip_page_numbers: bool = True      # ลบบรรทัดที่มีแต่ตัวเลขล้วน (เลขหน้าจาก PDF)


@dataclass
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    query_prefix: str = "query: "        # E5 บังคับต้องมี prefix
    passage_prefix: str = "passage: "
    normalize: bool = True               # normalize -> dot product == cosine
    batch_size: int = 16
    device: str = "cpu"
    max_seq_length: int = 512


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"        # "dense" | "bm25" | "hybrid"
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60             # ค่าคงที่ใน Reciprocal Rank Fusion
    top_k_final: int = 5        # จำนวน context ที่ส่งเข้า LLM
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 20  # จำนวน candidate ที่โยนเข้า cross-encoder


@dataclass
class GeneratorConfig:
    backend: str = "anthropic"   # "anthropic" | "openai" | "hf" | "echo"
    model: str = "claude-sonnet-5"
    temperature: float = 0.0     # RAG ต้องการ determinism ไม่ใช่ความสร้างสรรค์
    max_tokens: int = 1024
    hf_model: str = "scb10x/llama3.2-typhoon2-3b-instruct"


@dataclass
class EvalConfig:
    gold_path: str = "data/qa_gold.jsonl"
    metrics: List[str] = field(
        default_factory=lambda: ["bertscore", "rouge_l", "llm_judge",
                                 "ragas_faithfulness", "ragas_answer_relevance"]
    )
    bertscore_model: str = "xlm-roberta-large"
    judge_backend: str = "anthropic"
    judge_model: str = "claude-sonnet-5"
    ks: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class Config:
    name: str = "baseline"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ---------- helpers ----------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        from .env import load_dotenv
        load_dotenv()          # อ่าน .env ทุกครั้งที่โหลด config
        cfg = cls()
        # ถ้าไม่ระบุไฟล์ ให้ใช้ configs/baseline.yaml โดยอัตโนมัติ
        # (ไม่งั้นค่าที่ผู้ใช้แก้ในไฟล์ config จะถูกเมิน แล้วใช้ค่า default ในโค้ดแทน
        #  ซึ่งทำให้ "แก้ config แล้วไม่มีอะไรเปลี่ยน" — สับสนมาก)
        if path is None:
            default = ROOT / "configs" / "baseline.yaml"
            if default.exists():
                path = str(default)
                print(f"[config] ใช้ค่าจาก {default.name} (ระบุด้วย --config เพื่อเปลี่ยน)")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = cfg.merge(data)
        return cfg

    def merge(self, data: Dict[str, Any]) -> "Config":
        """merge dict ทับ config เดิม (ใช้ตอน ablation override)"""
        new = copy.deepcopy(self)
        for key, value in data.items():
            if not hasattr(new, key):
                raise KeyError(f"Unknown config key: {key}")
            attr = getattr(new, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                for k2, v2 in value.items():
                    if not hasattr(attr, k2):
                        raise KeyError(f"Unknown config key: {key}.{k2}")
                    setattr(attr, k2, v2)
            else:
                setattr(new, key, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    # ชื่อไฟล์ index ผูกกับ (chunk config + embedding model)
    # ถ้าเปลี่ยน chunk size แล้วลืม re-index จะได้ index คนละชุด -> กัน bug คลาสสิก
    @property
    def index_tag(self) -> str:
        m = self.embedding.model_name.split("/")[-1]
        c = self.chunk
        return f"{m}_c{c.max_tokens}_o{c.overlap_tokens}"
