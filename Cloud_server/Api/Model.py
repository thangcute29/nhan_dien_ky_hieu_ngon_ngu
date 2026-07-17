# cloud_server/api/models.py
from pydantic import BaseModel
from typing import Optional

class EdgeCaseUpload(BaseModel):
    label: Optional[str] = None   # Nhãn do người dùng đề xuất (có thể rỗng)