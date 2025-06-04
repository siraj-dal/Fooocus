from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from modules.simple_txt2img import generate_from_prompt
from modules.patch import patch_settings, PatchSettings
from PIL import Image
import os
import uuid
import numpy as np

# Colab & ngrok support
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

NGROK_TOKEN = "your_ngrok_token_here"  # Replace with your token

# FastAPI App
app = FastAPI()

# Patch initialization
pid = os.getpid()
if pid not in patch_settings:
    patch_settings[pid] = PatchSettings()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: PromptRequest):
    prompt = data.prompt
    print(f"🧠 Generating image from: {prompt}")
    
    image = generate_from_prompt(
        positive_prompt=prompt,
        width=1024,
        height=1024,
        steps=30
    )

    if isinstance(image, list) and isinstance(image[0], np.ndarray):
        pil_image = Image.fromarray(image[0])
        temp_file = f"/tmp/image_{uuid.uuid4().hex[:8]}.png"
        pil_image.save(temp_file)
        print(f"✅ Image generated and saved to {temp_file}")
        return FileResponse(temp_file, media_type="image/png")
    else:
        return JSONResponse(status_code=500, content={
            "error": "Unexpected image format",
            "type": str(type(image)),
            "value": str(image)[:300]
        })

# Run with or without ngrok
if __name__ == "__main__":
    import uvicorn

    if IN_COLAB:
        import nest_asyncio
        from pyngrok import ngrok

        nest_asyncio.apply()
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(8000)
        print(f"🚀 Public ngrok URL: {public_url}")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
