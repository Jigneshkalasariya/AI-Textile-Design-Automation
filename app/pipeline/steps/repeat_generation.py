from PIL import Image
from typing import Dict, Any
from app.models.request_models import RepeatGenerationConfig
from app.core.logger import logger

def generate_repeat(img: Image.Image, repeat_info: Dict[str, Any], config: RepeatGenerationConfig) -> Image.Image:
    """
    Creates a repeat pattern layout of the image.
    Supports:
        - straight: simple grid tiling
        - half-drop: alternate columns are shifted vertically by 50%
    """
    logger.info("Running Step 9: Repeat Generation")
    
    # Extract dimensions from detected repeat info or fall back to input image size
    repeat_w = repeat_info.get("repeat_width", img.width)
    repeat_h = repeat_info.get("repeat_height", img.height)
    repeat_type = config.repeat_type.lower()
    
    cols = config.horizontal_tiles
    rows = config.vertical_tiles
    
    # Crop the image to the actual detected repeat unit size
    tile = img.crop((0, 0, repeat_w, repeat_h))
    
    # Calculate output canvas dimensions
    canvas_w = cols * repeat_w
    # For half-drop, we need a bit of extra vertical space to handle the shifts
    extra_h = (repeat_h // 2) if repeat_type == "half-drop" else 0
    canvas_h = rows * repeat_h + extra_h
    
    # Create output canvas matching the source image mode (RGB, RGBA, P, etc.)
    canvas = Image.new(img.mode, (canvas_w, canvas_h))
    
    logger.debug(f"Tiling {cols}x{rows} repeat (type={repeat_type}, unit={repeat_w}x{repeat_h})")
    
    for c in range(cols):
        # Determine vertical shift for this column
        v_shift = 0
        if repeat_type == "half-drop" and (c % 2 == 1):
            v_shift = repeat_h // 2
            
        for r in range(rows + (1 if v_shift > 0 else 0)):
            x = c * repeat_w
            y = r * repeat_h + v_shift
            
            # Draw if it fits within canvas
            if y < canvas_h:
                canvas.paste(tile, (x, y))
                
    logger.info("Repeat tile grid generated successfully.")
    return canvas
