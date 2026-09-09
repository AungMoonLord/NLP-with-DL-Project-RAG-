"""STEP 6 — Generation layer (pluggable backend)

ออกแบบเป็น interface เดียว เพราะ:
  - ตอน dev/CI ใช้ backend "echo" ได้โดยไม่ต้องมี API key และไม่เสียเงิน
  - ตอน demo ใช้ Anthropic/OpenAI
  - ถ้าอาจารย์ถามว่า "รันแบบ offline ได้ไหม" -> ใช้ backend "hf" (Typhoon 3B บน CPU)
  - การประเมิน retrieval แยกออกจาก generation ได้ ทำให้ ablation ชี้ชัดว่าอะไรทำให้ดีขึ้น
"""
from __future__ import annotations

import os
from typing import List, Optional

from .config import GeneratorConfig
from .prompts import SYSTEM_PROMPT, ANSWER_TEMPLATE


class BaseGenerator:
    def generate(self, system: str, user: str, **kw) -> str:
        raise NotImplementedError

    def answer(self, question: str, context: str) -> str:
        return self.generate(SYSTEM_PROMPT,
                             ANSWER_TEMPLATE.format(context=context, question=question))


class EchoGenerator(BaseGenerator):
    """ไม่เรียก LLM จริง — ใช้ทดสอบ retrieval pipeline แบบ offline"""

    def generate(self, system: str, user: str, **kw) -> str:
        return "[ECHO MODE] ไม่ได้เรียก LLM จริง\n" + user[-1200:]


class AnthropicGenerator(BaseGenerator):
    def __init__(self, cfg: GeneratorConfig):
        import anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, system: str, user: str, **kw) -> str:
        resp = self.client.messages.create(
            model=kw.get("model", self.cfg.model),
            max_tokens=kw.get("max_tokens", self.cfg.max_tokens),
            temperature=kw.get("temperature", self.cfg.temperature),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAIGenerator(BaseGenerator):
    """ใช้ได้กับ OpenAI และ endpoint ที่ compatible (vLLM, Ollama, OpenTyphoon ฯลฯ)"""

    def __init__(self, cfg: GeneratorConfig):
        from openai import OpenAI
        from .env import load_dotenv, require
        load_dotenv()
        self.cfg = cfg
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(
            api_key=require("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=120.0,      # endpoint ของผู้ให้บริการรายเล็กมักช้ากว่า default 10 วิ
            max_retries=3,
        )
        print(f"[generator] OpenAI-compatible | base_url={base_url or 'default'} "
              f"| model={cfg.model}")

    def generate(self, system: str, user: str, **kw) -> str:
        resp = self.client.chat.completions.create(
            model=kw.get("model", self.cfg.model),
            temperature=kw.get("temperature", self.cfg.temperature),
            max_tokens=kw.get("max_tokens", self.cfg.max_tokens),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content


class HFGenerator(BaseGenerator):
    """โมเดลไทยขนาดเล็กรันเครื่องตัวเอง (ช้าบน CPU แต่ไม่ต้องพึ่ง API)"""

    def __init__(self, cfg: GeneratorConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model, torch_dtype=torch.float32, device_map="auto")

    def generate(self, system: str, user: str, **kw) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg.max_tokens,
            do_sample=self.cfg.temperature > 0, temperature=max(self.cfg.temperature, 1e-5))
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_generator(cfg: GeneratorConfig) -> BaseGenerator:
    backend = cfg.backend.lower()
    if backend == "anthropic":
        return AnthropicGenerator(cfg)
    if backend == "openai":
        return OpenAIGenerator(cfg)
    if backend == "hf":
        return HFGenerator(cfg)
    if backend == "echo":
        return EchoGenerator()
    raise ValueError(f"ไม่รู้จัก generator backend: {backend}")
