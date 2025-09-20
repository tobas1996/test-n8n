from typing import Dict, List, TypedDict
from threading import Lock

class ChatTurn(TypedDict):
    role: str   # "user" | "assistant"
    content: str

# Diccionario en RAM: session_id → historial de mensajes
_MEMORY: Dict[str, List[ChatTurn]] = {}
_LOCK = Lock()
MAX_TURNS = 4  # guardamos últimos 4 turnos (ajusta según quieras)

def get_memory(session_id: str) -> List[ChatTurn]:
    if not session_id:
        return []
    sid = str(session_id)
    with _LOCK:
        return list(_MEMORY.get(sid, []))

def append_to_memory(session_id: str, role: str, content: str) -> None:
    if not session_id:
        return
    sid = str(session_id)
    with _LOCK:
        convo = _MEMORY.setdefault(sid, [])
        convo.append({"role": role, "content": content})
        if len(convo) > MAX_TURNS:
            _MEMORY[sid] = convo[-MAX_TURNS:]

def reset_memory(session_id: str) -> None:
    """Elimina por completo la memoria asociada a un chat_id/session_id"""
    if not session_id:
        return
    sid = str(session_id)
    with _LOCK:
        _MEMORY.pop(sid, None)
