import os
import ssl
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

from modules.simple_txt2img import generate_from_prompt
from PIL import Image

def test_generation():
    from traceback import print_exc
    from PIL import Image
    import numpy as np

    # prompt = "a beautiful sunset over mountains, photorealistic, high quality"
    prompt = "Full-body image of a fashionable young man wearing urban streetwear, standing confidently, photorealistic style, neutral background, soft lighting, modern hairstyle, high detail."
    print(f"Generating image with prompt: {prompt}")
    
    try:
        image = generate_from_prompt(
            positive_prompt=prompt,
            width=1024,
            height=1024,
            steps=30
        )
        
        if isinstance(image, list) and isinstance(image[0], np.ndarray):
            pil_image = Image.fromarray(image[0])
            pil_image.save('test_output.png')
            print("✅ Image saved as 'test_output.png'")
        else:
            print("❌ Error: Unexpected image format")
            print("Returned type:", type(image))
            print("Returned value:", image)

    except Exception as e:
        print("❌ Exception occurred during generation:")
        print_exc()


if __name__ == '__main__':
    import os
    from modules.patch import patch_settings, PatchSettings

    # Initialize patch settings for the current process ID if not already done
    pid = os.getpid()
    if pid not in patch_settings:
        patch_settings[pid] = PatchSettings()


    test_generation()