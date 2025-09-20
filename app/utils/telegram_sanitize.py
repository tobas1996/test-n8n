# utils/telegram_sanitize.py
import re

MDV2_SPECIALS = r'[_*[\]()~`>#+\-=|{}.!]'
def escape_markdown_v2(text: str) -> str:
    if not text:
        return "…"
    # Reemplaza caracteres especiales de Markdown por sus equivalentes de ancho completo
    replacements = {
        "*": "✱",
        "_": "﹍",
        "`": "´",
        "[": "［",
        "]": "］",
        "(": "（",
        ")": "）"
    }
    for bad, safe in replacements.items():
        text = text.replace(bad, safe)
    # Opcional: quita tags HTML que algunos modelos meten
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()
