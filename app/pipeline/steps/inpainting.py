import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any
import torch
from app.models.request_models import InpaintingConfig
from app.core.config import settings
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

# Try importing diffusers
try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers/transformers not available. OpenCV standard inpainting will be used.")

def repair_pattern(img: Image.Image, repeat_info: Dict[str, Any], config: InpaintingConfig) -> Image.Image:
    """
    Applies inpainting to repair edges or missing regions.
    Creates an edge boundary mask (seams) and repairs it to make the pattern seamless.
    Tries Stable Diffusion inpainting if enabled and weights are available,
    otherwise falls back to OpenCV's fast marching inpainting (cv2.inpaint).
    """
    logger.info("Running Step 5: Pattern Repair (Inpainting)")
    
    width, height = img.size
    
    # 1. Create a seam border mask (16 pixels wide around the boundary)
    # This simulates fixing repeat edges to make them seamless
    mask = np.zeros((height, width), dtype=np.uint8)
    border = 16
    cv2.rectangle(mask, (0, 0), (width, height), 255, border)
    
    mask_pil = Image.fromarray(mask)

    if config.enabled and DIFFUSERS_AVAILABLE and settings.ALLOW_MODEL_DOWNLOADS:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Running SD Inpainting on device: {device}...")
            
            # Load Stable Diffusion or FLUX pipeline based on config
            # Path can be a local model folder or download path
            if config.model == "flux":
                logger.warning("FLUX inpainting not natively installed yet. Falling back to SD Inpainting or OpenCV.")
            model_id = "runwayml/stable-diffusion-inpainting"
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16
            )
            pipe = pipe.to(device)
            
            # Resize image/mask to 512x512 as SD standard size
            orig_size = img.size
            img_resized = img.resize((512, 512))
            mask_resized = mask_pil.resize((512, 512))
            
            result = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=img_resized,
                mask_image=mask_resized,
                strength=config.strength
            ).images[0]
            
            # Resize back
            logger.info("SD Inpainting completed successfully.")
            return result.resize(orig_size)
            
        except Exception as e:
            logger.error(f"{config.model.upper()} Inpainting failed: {e}. Falling back to OpenCV fast-marching.")

    # OpenCV Fallback (Fast Marching inpainting)
    logger.debug("Executing OpenCV boundary inpainting fallback...")
    cv_img = pil_to_cv2(img)
    
    # Check channels (BGR vs BGRA)
    if cv_img.shape[2] == 4:
        # Inpaint BGR channel, preserve Alpha
        bgr = cv_img[:, :, :3]
        alpha = cv_img[:, :, 3]
        # Inpaint mask regions
        inpainted_bgr = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
        # Re-assemble
        cv_img = cv2.merge([inpainted_bgr[:, :, 0], inpainted_bgr[:, :, 1], inpainted_bgr[:, :, 2], alpha])
    else:
        cv_img = cv2.inpaint(cv_img, mask, 7, cv2.INPAINT_TELEA)
        
    logger.info("OpenCV boundary repair completed.")
    return cv2_to_pil(cv_img)
