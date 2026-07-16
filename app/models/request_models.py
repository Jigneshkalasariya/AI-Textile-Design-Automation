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
        default=["PNG", "BMP", "TIFF", "SVG", "PSD"], 
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
🎨 3. COLOR RESTORATION
Restore faded colors (LAB color space preferred)
Normalize contrast
Balance brightness
Preserve original color intent
🔍 4. DETAIL ENHANCEMENT
Increase resolution (2x–4x upscale)
Remove blur
Improve sharpness
Preserve micro details
Enhance fine textures
✏️ 5. EDGE & LINE OPTIMIZATION
Strengthen line clarity
Smooth jagged edges
Maintain natural curves
Produce crisp, clean boundaries

🖤 ADDITIONAL OUTPUT (MANDATORY)
Black & White Sketch Design

Generate a separate output with:
Pure black lines on white background
No grayscale (binary or near-binary)
High edge fidelity
Clean continuous strokes
No loss of motif or repeat structure
Textile CAD-friendly line thickness

📦 OUTPUT VARIANTS (GENERATE ALL)
Version 1: Balanced Restoration
Natural enhancement
No aggressive sharpening
Version 2: Maximum Sharpness
Strong edge clarity
High-detail enhancement
Version 3: Print Optimized
CMYK-safe tones
Controlled contrast
Color stability for fabric printing
Version 4: Vectorization Optimized
Flat regions
Clean edges
Reduced noise for tracing
Version 5: Repeat Detection Optimized
Highlight repeat clarity
Seamless tiling preservation
Version 6: Texcelle Import Optimized
Perfect flat bitmap
Clean boundaries
CAD-ready precision

📁 EXPORT REQUIREMENTS
For EACH version + sketch:
Format: BMP
DPI: 600
Compression: None (Lossless)
Quality: Maximum
Resolution: High (minimum 4K or upscaled)
Background: Clean and uniform

📤 OUTPUT STRUCTURE
/output/
   v1_balanced.bmp
   v2_sharp.bmp
   v3_print.bmp
   v4_vector.bmp
   v5_repeat.bmp
   v6_texcelle.bmp
   sketch_bw.bmp"""

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
