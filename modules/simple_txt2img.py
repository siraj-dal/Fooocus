import modules.default_pipeline as pipeline
import modules.core as core
import modules.config
from modules.util import get_enabled_loras

def generate_from_prompt(positive_prompt, width=1024, height=1024, steps=30):
    """Generate an image from a text prompt using Fooocus pipeline.
    
    Args:
        positive_prompt (str): The text description of desired image
        width (int): Output image width (default: 1024)
        height (int): Output image height (default: 1024)
        steps (int): Number of sampling steps (default: 30)
    
    Returns:
        PIL.Image: The generated image
    """
    # Initialize models if needed
    pipeline.refresh_everything(
        refiner_model_name=modules.config.default_refiner_model_name,
        base_model_name=modules.config.default_base_model_name,
        loras=get_enabled_loras(modules.config.default_loras),
        vae_name=modules.config.default_vae
    )
    
    # Encode text prompt
    positive_cond = pipeline.clip_encode([positive_prompt])
    negative_cond = pipeline.clip_encode([''])
    
    # Generate image
    image = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=steps,
        switch=steps,
        width=width,
        height=height,
        image_seed=123456,
        callback=None,
        sampler_name='dpmpp_2m',
        scheduler_name='karras',
        denoise=1.0,
        tiled=False,
        cfg_scale=7.0
    )
    
    return image