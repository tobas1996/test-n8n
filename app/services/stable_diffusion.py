from __future__ import annotations

import torch
import base64
from io import BytesIO
from PIL import Image
from fastapi import Request
from app.models.requests import GenerateImageRequest
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
CONFIG = {
    "default_model": "E:/AI/models/txt2image/realisticVisionV60B1_v51HyperVAE.safetensors",
    "default_scheduler": "DPM++ 2M Karras",
    "force_gpu": True,
    "enable_xformers": False,          # Desactivado por defecto para evitar error si no está instalado
    "enable_attention_slicing": True,
    "enable_cpu_offload": False,
    "disable_safety_checker": True,
    "vram_limit_gb": 8,
    "safe_mode": True,
    "max_resolution_high_vram": 1024,
    "max_resolution_low_vram": 768,
    "cache_pipeline": True,
}

# Guardamos solo un pipeline en memoria
_CURRENT_PIPELINE: StableDiffusionPipeline | None = None
_CURRENT_MODEL_PATH: str | None = None


# =========================
# DETECTAR GPU Y VRAM
# =========================
def detect_device() -> str:
    return "cuda" if CONFIG["force_gpu"] and torch.cuda.is_available() else "cpu"


def get_vram_gb() -> float:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / (1024**3), 2)
    return 0.0


# =========================
# UTILIDADES
# =========================
def clamp_resolution(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= max_side and height <= max_side:
        return width, height
    scale = min(max_side / width, max_side / height)
    new_w = int(round(width * scale / 8)) * 8
    new_h = int(round(height * scale / 8)) * 8
    return max(64, new_w), max(64, new_h)


# =========================
# CARGA DEL PIPELINE
# =========================
def load_pipeline(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Cargando modelo desde archivo local: {model_path}")

    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True
    ).to(device)

    return pipe


def get_or_load_pipeline(model_path: str) -> StableDiffusionPipeline:
    global _CURRENT_PIPELINE, _CURRENT_MODEL_PATH

    if _CURRENT_MODEL_PATH == model_path and _CURRENT_PIPELINE is not None:
        print(f"[INFO] Usando modelo ya cargado: {model_path}")
        return _CURRENT_PIPELINE

    # Descargar el modelo previo
    if _CURRENT_PIPELINE is not None:
        print(f"[INFO] Liberando modelo anterior: {_CURRENT_MODEL_PATH}")
        del _CURRENT_PIPELINE
        torch.cuda.empty_cache()

    # Cargar el nuevo
    _CURRENT_PIPELINE = load_pipeline(model_path)
    _CURRENT_MODEL_PATH = model_path
    return _CURRENT_PIPELINE


# =========================
# SCHEDULERS
# =========================
def set_scheduler(pipe: StableDiffusionPipeline, scheduler_name: str) -> None:
    name = (scheduler_name or "").lower().strip()

    if name in ("dpmsolvermultistepscheduler", "dpm solver", "dpmsolver"):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif name in ("euler a", "eulerancestraldiscretescheduler", "euler ancestral", "euler_ancestral"):
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    elif name in ("dpm++ 2m karras", "dpmpp 2m karras", "dpmpp2m karras"):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
        )
    else:
        print(f"[WARN] Scheduler '{scheduler_name}' no reconocido. Usando DPMSolverMultistepScheduler.")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# =========================
# GENERACIÓN DE IMAGEN
# =========================
def generate_image(data: GenerateImageRequest, request: Request) -> str:
    device = detect_device()
    vram_gb = get_vram_gb()

    max_side = (
        CONFIG["max_resolution_low_vram"]
        if (CONFIG["safe_mode"] and vram_gb <= CONFIG["vram_limit_gb"])
        else CONFIG["max_resolution_high_vram"]
    )

    orig_w, orig_h = data.width, data.height
    data.width, data.height = clamp_resolution(data.width, data.height, max_side)
    if (data.width, data.height) != (orig_w, orig_h):
        print(f"[INFO] Resolución ajustada {orig_w}x{orig_h} → {data.width}x{data.height} (VRAM {vram_gb} GB)")

    model_path = data.model_path or CONFIG["default_model"]
    pipe = get_or_load_pipeline(model_path)
    set_scheduler(pipe, data.scheduler or CONFIG["default_scheduler"])

    generator = torch.Generator(device).manual_seed(
        data.seed if data.seed != -1 else torch.seed()
    )

    def check_cancel(_pipe, _step_index, _timestep, callback_kwargs):
        try:
            import anyio
            disconnected = anyio.from_thread.run(request.is_disconnected)
        except Exception:
            disconnected = False
        if disconnected:
            print("[INFO] Cliente canceló la petición.")
            raise RuntimeError("Generación cancelada por el cliente")
        return callback_kwargs

    result = pipe(
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        width=data.width,
        height=data.height,
        num_inference_steps=data.steps,
        guidance_scale=data.guidance_scale,
        generator=generator,
        callback_on_step_end=check_cancel,
    )
    image: Image.Image = result.images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] VRAM temporal liberada")

    return img_b64
