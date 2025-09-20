# app/services/user_styles.py
from typing import Dict, Optional

_user_styles: Dict[str, str] = {}

def set_style(chat_id: str, style: str) -> None:
    _user_styles[str(chat_id)] = style

def pop_style(chat_id: str) -> Optional[str]:
    # devuelve y ELIMINA el estilo (uso de una sola vez)
    return _user_styles.pop(str(chat_id), None)

def has_style(chat_id: str) -> bool:
    return str(chat_id) in _user_styles

