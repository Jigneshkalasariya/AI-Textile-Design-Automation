import colorsys
from PIL import Image
from typing import List, Tuple, Dict
from app.models.request_models import ColorVariantsConfig
from app.core.logger import logger

def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def generate_color_variants(
    quantized_img: Image.Image, 
    base_palette: List[Tuple[int, int, int]], 
    config: ColorVariantsConfig
) -> Dict[str, Image.Image]:
    """
    Generates alternative colorways (color variants) using color harmony principles
    applied in the HSV color space.
    Returns:
        - Dict mapping colorway name to recolored indexed PIL Image.
    """
    logger.info("Running Step 10: Color Variants")
    
    variants = {}
    if not config.enabled or not base_palette:
        logger.debug("Color variants generation disabled or base palette is empty.")
        return variants

    # Max variants requested
    limit = config.variant_count
    
    # We will generate variants based on harmony types
    harmonies = config.harmony_types[:limit]
    
    # If config harmony types list is empty, default to complementary/analogous
    if not harmonies:
        harmonies = ["complementary", "analogous", "triad"][:limit]

    # Padding function to pad palette to 768 elements (256 RGB colors) for PIL 'P' mode
    def make_pil_palette(rgb_list: List[Tuple[int, int, int]]) -> List[int]:
        flat = []
        for rgb in rgb_list:
            flat.extend(rgb)
        # Pad with zeros
        flat.extend([0] * (768 - len(flat)))
        return flat

    for harmony in harmonies:
        variant_palette = []
        
        for rgb in base_palette:
            h, s, v = rgb_to_hsv(rgb)
            
            if harmony == "complementary":
                # Shift hue by 180 degrees (0.5 in 0.0-1.0 range)
                h = (h + 0.5) % 1.0
            elif harmony == "analogous":
                # Shift hue by 30 degrees (approx 0.083 in 0.0-1.0 range)
                h = (h + 0.083) % 1.0
                # slightly boost saturation
                s = min(1.0, s * 1.1)
            elif harmony == "triad":
                # Shift hue by 120 degrees (0.33)
                h = (h + 0.33) % 1.0
            elif harmony == "monochromatic":
                # Shift brightness/saturation, keep hue
                s = min(1.0, s * 0.8)
                v = min(1.0, v * 1.2)
            else:
                # Default random shift
                h = (h + 0.15) % 1.0

            variant_palette.append(hsv_to_rgb((h, s, v)))

        # Create a new image using the same indexed pixel data but a new palette
        recolored_img = quantized_img.copy()
        recolored_img.putpalette(make_pil_palette(variant_palette))
        
        variant_name = f"colorway_{harmony}"
        variants[variant_name] = recolored_img
        logger.debug(f"Generated colorway variant: {variant_name}")

    logger.info(f"Successfully generated {len(variants)} colorway variants.")
    return variants
