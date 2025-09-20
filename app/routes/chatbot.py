from fastapi import APIRouter, HTTPException, Query
from app.models.requests import GenerateChatRequest, ResetRequest
from app.models.responses import GenerateChatResponse
from app.services.ollama_client import ollama_generate
from app.services.memory import get_memory, append_to_memory, reset_memory

router = APIRouter()

DEFAULT_SYSTEM = (
    "You are an advanced conversational assistant whose sole purpose is to interact with the user entirely in Spanish. "
    "You must always sound natural, coherent, and fluent in Spanish, as if you were a native speaker. "
    "You are expressive, creative, and capable of adapting your tone: you can be friendly, sarcastic, dramatic, humorous, or professional, depending on context. "
)

STYLE_RULES = (
    "[STYLES]\n"
    "Keep your answers proportional to the conversation: short when the user is brief, longer when the context is rich.\n"
)

IMPORTANT_RULES = (
    "[IMPORTANT RULES]\n"
    "- You must always output at least one sentence. Never return an empty response.\n"
    "- Always respond in Spanish, without exceptions.\n"
    "- Do NOT include English words or phrases.\n"
    "- Do NOT provide translations in parentheses or outside them.\n"
    "- Do NOT explain meanings in other languages.\n"
    "- Write naturally, fluently, and coherently as a native Spanish speaker.\n"
    "- Maintain the requested style (epic, sarcastic, friendly, etc.), but always in Spanish.\n"
)

def compose_prompt(
    system_prompt: str, history: list, user_prompt: str, max_turns: int = 12
) -> str:
    """
    Build a complete prompt for Ollama with:
    - system instructions
    - conversation history (last N turns)
    - user input
    - assistant role
    - important rules (global constraints)
    """
    lines = [f"[SYSTEM]\n{system_prompt}\n"]

    if history:
        lines.append("[HISTORY]")
        for turn in history[-max_turns:]:
            role = turn["role"].upper()
            lines.append(f"{role}: {turn['content']}")
        lines.append("")  # newline

    lines.append("[USER]")
    lines.append(user_prompt)
    lines.append("\n[ASSISTANT]\n")

    # Add global constraints at the end
    lines.append(IMPORTANT_RULES)
    lines.append(STYLE_RULES)

    return "\n".join(lines)

@router.post("/generate-chat", response_model=GenerateChatResponse)
def generate_chat(req: GenerateChatRequest):
    try:
        # 1) Recuperar historial si hay session_id
        history = get_memory(req.session_id) if req.session_id else []

        # 2) Construir prompt compuesto
        system = req.system_prompt or DEFAULT_SYSTEM
        composed_prompt = compose_prompt(system, history, req.prompt)

        # 3) Ajuste dinámico de num_predict
        base = 64
        bonus = min(len(req.prompt) // 20, 64)  # más tokens si el prompt del usuario es largo
        dynamic_num_predict = base  # podrías usar base+bonus si quieres aprovecharlo

        # 4) Llamar a Ollama
        payload = {
            "model": req.model,
            "prompt": composed_prompt,
            "temperature": req.temperature,
            "num_predict": dynamic_num_predict,
            "stream": False,
            "repeat_penalty": req.repeat_penalty,
            "top_k": req.top_k,
            "top_p": req.top_p,
        }
        data = ollama_generate(payload)
        answer = data.get("response", "")
        if not answer:
            answer = "…"  # fallback genérico
        done = data.get("done", True)

        # 5) Guardar turno en memoria
        if req.session_id:
            append_to_memory(req.session_id, "user", req.prompt)
            append_to_memory(req.session_id, "assistant", answer)

        return GenerateChatResponse(response=answer, done=done)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-chat")
def reset_chat(req: ResetRequest):
    """
    Endpoint para limpiar la memoria de conversación de un chat concreto.
    Espera un JSON con { "session_id": "<chat_id>" }
    """
    try:
        reset_memory(req.session_id)
        return {"status": "ok", "message": f"Memoria del chat {req.session_id} eliminada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
