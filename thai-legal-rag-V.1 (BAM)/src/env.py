"""อ่านไฟล์ .env เข้าสู่ environment variables

เขียนเองแทนการใช้ python-dotenv เพื่อไม่เพิ่ม dependency
กติกา: ค่าที่ตั้งไว้ใน environment อยู่แล้ว จะไม่ถูกเขียนทับโดยไฟล์ .env
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None, override: bool = False) -> int:
    """คืนจำนวนตัวแปรที่โหลดเข้ามา"""
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return 0

    loaded = 0
    # อ่านเป็น utf-8-sig เพื่อรองรับไฟล์ที่ Notepad บันทึกพร้อม BOM
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def require(name: str, hint: str = "") -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"ไม่พบตัวแปร {name}\n"
            f"วิธีแก้: สร้างไฟล์ชื่อ .env ไว้ในโฟลเดอร์ thai-legal-rag แล้วใส่บรรทัด\n"
            f"    {name}=ค่าของคุณ\n" + (f"หมายเหตุ: {hint}" if hint else "")
        )
    return val
