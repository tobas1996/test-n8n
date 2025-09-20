from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print(">>> Cargando modelo BLIP (esto puede tardar la primera vez)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print(">>> Modelo cargado!")

# Cambia esta ruta a una imagen que sepas que existe en la carpeta
img_path = "00001-3160653179.png"

print(f">>> Abriendo imagen: {img_path}")
image = Image.open(img_path).convert("RGB")

print(">>> Procesando imagen...")
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("👉 Caption generado:", caption)
