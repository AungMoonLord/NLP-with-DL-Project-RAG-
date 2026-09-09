"""ทดสอบการเชื่อมต่อ LLM API ก่อนรันงานจริง

    python scripts/test_api.py                    # ใช้ model จาก config
    python scripts/test_api.py --model typhoon-v2-70b-instruct
    python scripts/test_api.py --list             # ขอรายชื่อ model ที่ endpoint รองรับ

รันตัวนี้ให้ผ่านก่อนเสมอ จะได้ไม่เสียเวลารัน ingest 10 นาทีแล้วมาพังตอนเรียก API
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.env import load_dotenv


def list_models():
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL") or None)
    print("[api] รายชื่อ model ที่ endpoint นี้รองรับ:")
    for m in client.models.list().data:
        print("   -", m.id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    n = load_dotenv()
    print(f"[api] โหลดตัวแปรจาก .env จำนวน {n} ตัว")

    import os
    key = os.environ.get("OPENAI_API_KEY", "")
    url = os.environ.get("OPENAI_BASE_URL", "(ไม่ได้ตั้ง - จะใช้ของ OpenAI)")
    if not key:
        print("❌ ไม่พบ OPENAI_API_KEY  -> สร้างไฟล์ .env ตามตัวอย่างใน .env.example")
        sys.exit(1)
    # แสดงแค่ 4 ตัวแรก/ท้าย เพื่อยืนยันว่าอ่านถูกไฟล์ โดยไม่เปิดเผย key
    print(f"[api] OPENAI_API_KEY  = {key[:4]}...{key[-4:]}  (ความยาว {len(key)})")
    print(f"[api] OPENAI_BASE_URL = {url}")

    if args.list:
        list_models()
        return

    cfg = Config.load(ROOT / args.config)
    if args.model:
        cfg.generator.model = args.model
    if "MODEL_NAME_HERE" in cfg.generator.model:
        print("\n❌ ยังไม่ได้ตั้งชื่อ model")
        print("   แก้ใน configs/*.yaml ตรง generator.model และ evaluation.judge_model")
        print("   ถ้าไม่รู้ว่าใช้ชื่ออะไร ให้รัน: python scripts/test_api.py --list")
        sys.exit(1)

    from src.generator import build_generator
    gen = build_generator(cfg.generator)
    print("\n[api] กำลังส่งคำถามทดสอบ ...")
    out = gen.generate("ตอบสั้น ๆ เป็นภาษาไทย",
                       "ตอบกลับด้วยข้อความว่า เชื่อมต่อสำเร็จ เท่านั้น")
    print(f"[api] ได้รับคำตอบ: {out.strip()[:200]}")
    print("\n✅ เชื่อมต่อ API สำเร็จ พร้อมใช้งาน")


if __name__ == "__main__":
    main()
