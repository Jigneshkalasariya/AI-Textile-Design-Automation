from PIL import Image
from typing import List, Tuple
from app.models.request_models import ColorReductionConfig
from app.core.logger import logger

def reduce_colors(img: Image.Image, config: ColorReductionConfig) -> Tuple[Image.Image, List[Tuple[int, int, int]]]:
    """
    Reduces the image to a fixed palette of size `palette_size` (quantization).
    Optionally applies Floyd-Steinberg dithering for smooth color transitions.
    Returns:
        - Quantized PIL Image (in 'P' mode with internal palette)
        - List of (R, G, B) tuples representing the active palette
    """
    logger.info("Running Step 7: Color Reduction")
    
    # PIL's quantize needs an RGB image
    rgb_img = img.convert("RGB")
    
    # Perform quantization
    dither_val = Image.Dither.FLOYDSTEINBERG if config.dither else Image.Dither.NONE
    quantized_img = rgb_img.quantize(
        colors=config.palette_size,
        method=Image.Quantize.MEDIANCUT,
        dither=dither_val
    )
    
    # Extract palette colors
    # getpalette() returns a flat list of integers [R0, G0, B0, R1, G1, B1, ...]
    flat_palette = quantized_img.getpalette()
    
    # Chunk the flat list into RGB tuples, up to palette_size
    palette = []
    if flat_palette:
        for i in range(0, min(config.palette_size * 3, len(flat_palette)), 3):
            palette.append((flat_palette[i], flat_palette[i+1], flat_palette[i+2]))
            
    logger.info(f"Color reduction completed. Palette size: {len(palette)}")
    return quantized_img, palette
