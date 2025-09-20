from pydantic import BaseModel
from typing import Optional, Union

class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1
    scheduler: str = "DPMSolverMultistepScheduler"
    model_path: str = "E:/AI/models/txt2image/realisticVisionV60B1_v51HyperVAE.safetensors"
    style: Optional[str] = None
    session_id: Optional[Union[str, int]] = None

# --- Chatbot models (nuevo) ---
class GenerateChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.8
    num_predict: int = 128
    stream: bool = False
    repeat_penalty: float = 1.1
    top_k: int = 40
    top_p: float = 0.9

    # NEW: session-based memory (e.g., Telegram chat.id)
    session_id: Optional[Union[str, int]] = None

    # OPTIONAL: base instruction for the assistant
    system_prompt: Optional[str] = None


class SetStyleRequest(BaseModel):
    chat_id: Union[str, int]
    style: str


class ResetRequest(BaseModel):
    session_id: str