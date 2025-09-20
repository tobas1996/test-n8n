from pydantic import BaseModel

class GenerateImageResponse(BaseModel):
    image_base64: str

class GenerateChatResponse(BaseModel):
    response: str
    done: bool = True