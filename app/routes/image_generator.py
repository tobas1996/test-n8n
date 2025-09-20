from fastapi import APIRouter, Request, Query
from app.models.requests import GenerateImageRequest
from app.services.stable_diffusion import generate_image
from app.utils.styles_sd import STYLES
from app.models.requests import SetStyleRequest
from app.services.user_styles import has_style, pop_style, set_style


router = APIRouter()

@router.post("/generate")
def generate(req: GenerateImageRequest, request: Request):
    chat_id = str(req.session_id) if req.session_id else None

    # 🔑 Consumir el estilo una sola vez
    style = pop_style(chat_id) if chat_id else None

    if style and style in STYLES:
        sp = STYLES[style]
        # priorizamos la parte del usuario
        user_prompt = f"({req.prompt}:1.5)" if req.prompt else ""
        req.prompt = f"{user_prompt}, {sp['positive']}".strip(", ")
        req.negative_prompt = (
            (req.negative_prompt or "") + ", " + sp['negative']
        ).strip(", ")


    img_str = generate_image(req, request)
    return {"image_base64": img_str}

@router.post("/set-style")
def set_style_endpoint(req: SetStyleRequest):
    if req.style not in STYLES:
        return {"ok": False, "error": "Estilo no reconocido"}
    set_style(req.chat_id, req.style)
    return {"ok": True, "style": req.style}

@router.get("/get-style")
def get_style(chat_id: str = Query(...)):
    return {"has_style": has_style(chat_id)}