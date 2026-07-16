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

DEFAULT_CAD_PROMPT = """🚀 ANTIGRAVITY TEXTILE CAD PREPROCESSING PROMPT (FINAL VERSION)

You are a professional Textile CAD image preprocessing engine designed for production-ready Texcelle import pipelines.

Your task is to enhance and clean customer artwork without altering its original design identity.

🔒 STRICT PRESERVATION RULES (MANDATORY)

You MUST preserve exactly:
Every motif
Every flower
Every border
Every curve
Every line
Every color (only restore, do NOT change palette)
Every texture
Overall composition
Repeat boundaries

You MUST NOT:
Add new elements
Remove any design part
Distort repeat structure
Modify textile geometry
Stylize or redesign artwork

⚙️ PROCESSING PIPELINE
🧹 1. CLEANING
Remove JPEG compression artifacts
Remove scanning defects
Remove dust and micro-noise
Remove scratches
Remove unwanted background noise
📐 2. GEOMETRY CORRECTION
Correct perspective distortion
Fix skew and alignment
Ensure repeat boundaries remain intact
Maintain original proportions
🎨 3. COLOR & DETAIL ENHANCEMENT
Restore faded colors (LAB color space preferred)
Normalize contrast and brightness
Remove blur
Apply controlled sharpening
Preserve fine details and textures
✏️ 4. EDGE & LINE OPTIMIZATION
Strengthen line clarity
Smooth jagged edges
Maintain natural curves
Produce crisp, clean boundaries

📦 OUTPUT VARIANTS (GENERATE ALL)
1. MASTER ENHANCED IMAGE
High-quality processed version of original artwork

2. BLACK & WHITE SKETCH OUTPUT
Pure black lines on white background
No grayscale (binary or near-binary)
High edge accuracy
Preserve motifs and repeat structure
Textile CAD-friendly line thickness

3. COLOR VARIANT A (SOFT ENHANCED)
Natural textile-friendly tones
Balanced contrast
Subtle visual improvement

4. COLOR VARIANT B (VIBRANT ENHANCED)
Rich and enhanced colors
Higher contrast
Visually appealing while preserving design

📁 EXPORT REQUIREMENTS
For ALL outputs:
Format: BMP and PNG (both required)
DPI: 600
BMP: No compression (lossless)
PNG: Lossless compression
Quality: Maximum
Resolution: High (after upscale)
Background: Clean and uniform

📤 OUTPUT STRUCTURE
/output/
   master_enhanced.bmp
   master_enhanced.png
   sketch_bw.bmp
   sketch_bw.png
   color_variant_soft.bmp
   color_variant_soft.png
   color_variant_vibrant.bmp
   color_variant_vibrant.png"""

class TextileCADEngineConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Textile CAD Preprocessing Engine")
    prompt: str = Field(default=DEFAULT_CAD_PROMPT, description="Textile CAD Engine Prompt")
    dpi: int = Field(default=600, description="Target DPI resolution for export")
    generate_six_versions: bool = Field(default=True, description="Generate and export the 6 independent versions")

class ProcessConfig(BaseModel):
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
