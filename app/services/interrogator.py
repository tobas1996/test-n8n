from PIL import Image
from clip_interrogator import Config, Interrogator

print(">>> Cargando CLIP Interrogator (esto tarda un poco la primera vez)...")
config = Config()
ci = Interrogator(config)

# Cambia esto por tu imagen
img_path = "57326955-lovely-happy-woman-jumping-of-joy.jpg"
image = Image.open(img_path).convert("RGB")

print(f">>> Analizando {img_path} ...")
caption = ci.interrogate_fast(image)

print("👉 Prompt generado por CLIP Interrogator:")
print(caption)
