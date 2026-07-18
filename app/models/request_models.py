from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class EnhanceConfig(BaseModel):
    denoise_strength: float = Field(default=3.0, description="Denoise filter strength (0.0 to 10.0)")
    sharpen_strength: float = Field(default=1.5, description="Sharpening filter scaling factor")
    use_realesrgan: bool = Field(default=False, description="Enable RealESRGAN super-resolution")

class BackgroundRemovalConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable background removal")
    model_name: str = Field(default="sam2", description="Model name (sam2, rembg, etc.)")
    use_grounding_dino: bool = Field(default=False, description="Use Grounding DINO with SAM2")
    alpha_matting: bool = Field(default=False, description="Use alpha matting for soft edges")

class ObjectDetectionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable object detection for isolating motifs")
    model: str = Field(default="yolo11", description="Model (yolo11, grounding_dino)")
    confidence: float = Field(default=0.25, description="Confidence threshold")

class PatternDetectionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable automated repeat type and boundary detection")
    search_grid_size: int = Field(default=64, description="Step size for boundary search grid")

class InpaintingConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable pattern repair (inpainting)")
    model: str = Field(default="flux", description="Inpainting model to use (flux, sd)")
    prompt: str = Field(default="seamless matching textile pattern fabric", description="Prompt for inpainting")
    negative_prompt: str = Field(default="seams, lines, bad quality, blurry", description="Negative prompt")
    strength: float = Field(default=0.8, description="Inpainting strength")

class ColorSeparationConfig(BaseModel):
    num_colors: int = Field(default=8, description="Number of primary color layers to extract (KMeans)")
    color_space: str = Field(default="LAB", description="Color space for clustering (LAB or RGB)")

class ColorReductionConfig(BaseModel):
    palette_size: int = Field(default=8, description="Target palette size (indexed color reduction)")
    dither: bool = Field(default=True, description="Apply Floyd-Steinberg dithering")

class VectorizationConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable SVG path tracing")
    method: str = Field(default="potrace", description="Vectorization method (potrace, opencv)")
    simplify_tolerance: float = Field(default=1.0, description="Tolerance for contour approximation (Douglas-Peucker)")

class RepeatGenerationConfig(BaseModel):
    repeat_type: str = Field(default="straight", description="Repeat alignment ('straight', 'half-drop')")
    horizontal_tiles: int = Field(default=3, description="Number of tiles to output horizontally")
    vertical_tiles: int = Field(default=3, description="Number of tiles to output vertically")

class ColorVariantsConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable colorway generation")
    variant_count: int = Field(default=3, description="Number of alternative colorways to generate")
    harmony_types: List[str] = Field(
        default=["complementary", "analogous", "triad"], 
        description="Color harmony rules to apply"
    )

class OutputGenerationConfig(BaseModel):
    formats: List[str] = Field(
        default=["PNG", "SVG"], 
        description="Target formats to output"
    )

DEFAULT_CAD_PROMPT = """TEXTILE CAD AND TEXCELLE IMAGE PREPROCESSING ENGINE

Process the uploaded textile artwork into production-ready Texcelle files without changing its design identity.

Every output must have exactly the same pixel width, pixel height, aspect ratio, orientation, crop, and canvas boundaries as the uploaded image. Do not upscale, downscale, crop, extend, tile, rotate, or recompose the artwork.

Preserve every motif, flower, leaf, ornament, border, curve, outline, internal line, relative position, spacing, proportion, symmetry, repeat boundary, edge continuity, legitimate background, fine detail, texture, original color identity, and palette relationship.

Never add, invent, remove, regenerate, redesign, stylize, move, duplicate, or hallucinate design content. Never change motif geometry, repeat structure, or legitimate design backgrounds. Do not add grids, labels, watermarks, shadows, frames, or text.

Use the original uploaded image as the independent source for every output. Apply conservative edge-preserving cleaning only to defects that are clearly not artwork. Preserve geometry and canvas dimensions. Restore faded colors with LAB luminance correction while preserving hues and avoiding clipping. Apply controlled sharpening without halos.

Generate exactly four variants:
1. master_enhanced: naturally restored original palette and texture with balanced contrast and controlled sharpness.
2. sketch_bw: CAD line drawing containing only black 0 and white 255, with no grayscale, transparency, shading, hatching, gradients, or texture.
3. color_variant_soft: subtle contrast and color restoration without recoloring.
4. color_variant_vibrant: moderate contrast and saturation while preserving original color identities and avoiding clipping.

Export each variant independently as uncompressed BMP and lossless PNG at exactly the original pixel dimensions with 600 x 600 DPI metadata and no transparency."""

class TextileCADEngineConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Textile CAD Preprocessing Engine")
    prompt: str = Field(default=DEFAULT_CAD_PROMPT, description="Textile CAD Engine Prompt")
    dpi: int = Field(default=600, description="Target DPI resolution for export")
    generate_six_versions: bool = Field(default=False, description="Deprecated legacy output switch")

class ProcessConfig(BaseModel):
    cad_only: bool = Field(
        default=True,
        description="Generate only the eight production Texcelle files from the stored uploaded source",
    )
    cad_engine: TextileCADEngineConfig = Field(default_factory=TextileCADEngineConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)
    background_removal: BackgroundRemovalConfig = Field(default_factory=BackgroundRemovalConfig)
    object_detection: ObjectDetectionConfig = Field(default_factory=ObjectDetectionConfig)
    pattern_detection: PatternDetectionConfig = Field(default_factory=PatternDetectionConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    color_separation: ColorSeparationConfig = Field(default_factory=ColorSeparationConfig)
    color_reduction: ColorReductionConfig = Field(default_factory=ColorReductionConfig)
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)
    repeat_generation: RepeatGenerationConfig = Field(default_factory=RepeatGenerationConfig)
    color_variants: ColorVariantsConfig = Field(default_factory=ColorVariantsConfig)
    output_generation: OutputGenerationConfig = Field(default_factory=OutputGenerationConfig)
