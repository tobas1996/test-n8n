from fastapi import FastAPI
from app.routes import image_generator, chatbot

app = FastAPI(
    title="Stable Diffusion + Ollama API",
    version="1.0.0",
    description="API para generación de imágenes y chatbot, con Ollama y Stable Diffusion."
)

# Endpoint para generación de imágenes directas (SD puro)
app.include_router(image_generator.router, prefix="", tags=["Generate"])
app.include_router(chatbot.router, prefix="")