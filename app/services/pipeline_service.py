import os
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from PIL import Image

from app.core.logger import logger
from app.models.request_models import ProcessConfig
from app.services.storage_service import storage_service
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

# Import step modules (we'll implement them next)
from app.pipeline.steps.enhance import enhance_image
from app.pipeline.steps.background_removal import remove_background
from app.pipeline.steps.object_detection import detect_objects
from app.pipeline.steps.pattern_detection import detect_pattern
from app.pipeline.steps.inpainting import repair_pattern
from app.pipeline.steps.color_separation import separate_colors
from app.pipeline.steps.color_reduction import reduce_colors
from app.pipeline.steps.vectorization import vectorize_contours
from app.pipeline.steps.repeat_generation import generate_repeat
from app.pipeline.steps.color_variants import generate_color_variants
from app.pipeline.steps.output_generation import generate_outputs

class PipelineService:
    @staticmethod
    def run_pipeline(
        file_id: str,
        filename: str,
        config: ProcessConfig,
        progress_callback: Optional[Callable[[float, str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs the 11-step textile AI pipeline sequentially on the source image.
        Uses progress_callback to report state back to Celery.
        Saves intermediate and final outputs using the storage service under the job/file_id.
        """
        logger.info(f"Starting pipeline processing for file {filename} (ID: {file_id})")
        
        # Helper to trigger callbacks
        def update_progress(percent: float, step: str, msg: str):
            logger.info(f"[{file_id}] Step '{step}' - {percent}%: {msg}")
            if progress_callback:
                progress_callback(percent, step, msg)

        # 0. Load source image from storage
        update_progress(0.0, "load_input", "Loading input image from storage...")
        local_input_path = storage_service.get_local_path(file_id, filename)
        current_img = Image.open(local_input_path).convert("RGB")
        
        # Keep track of outputs to return
        pipeline_data = {
            "original": current_img,
            "job_id": file_id
        }

        # Step 1: Image Enhancement
        if config.cad_engine.enabled:
            update_progress(5.0, "cad_preprocessing", "Running Textile CAD Preprocessing (6 versions)...")
            from app.pipeline.steps.cad_engine import preprocess_image_cad
            cad_versions = preprocess_image_cad(current_img, config.cad_engine)
            pipeline_data["cad_versions"] = cad_versions
            if "version1" in cad_versions:
                current_img = cad_versions["version1"]

        update_progress(10.0, "enhance", "Enhancing image (denoising and sharpening)...")
        enhanced_img = enhance_image(current_img, config.enhance)
        pipeline_data["enhanced"] = enhanced_img
        current_img = enhanced_img

        # Step 2: Background Removal
        update_progress(20.0, "background_removal", "Removing background mask...")
        no_bg_img = remove_background(current_img, config.background_removal)
        pipeline_data["no_background"] = no_bg_img
        current_img = no_bg_img

        # Step 3: Object/Motif Detection
        update_progress(30.0, "object_detection", "Detecting motifs and object boundaries...")
        motifs, bboxes = detect_objects(current_img, config.object_detection)
        pipeline_data["motifs"] = motifs
        pipeline_data["bboxes"] = bboxes

        # Step 4: Pattern Detection (Boundaries and repeat type)
        update_progress(40.0, "pattern_detection", "Analyzing pattern structure and repeat boundaries...")
        repeat_info = detect_pattern(current_img, config.pattern_detection)
        pipeline_data["repeat_info"] = repeat_info

        # Step 5: Pattern Repair (Inpainting)
        update_progress(50.0, "inpainting", "Applying inpainting to repair boundaries...")
        # Inpaint using computed repeat boundaries or custom mask
        repaired_img = repair_pattern(current_img, repeat_info, config.inpainting)
        pipeline_data["repaired"] = repaired_img
        current_img = repaired_img

        # Step 6: Color Separation
        update_progress(60.0, "color_separation", "Performing color separation (layer segmentation)...")
        layers, separated_colors = separate_colors(current_img, config.color_separation)
        pipeline_data["layers"] = layers
        pipeline_data["separated_colors"] = separated_colors

        # Step 7: Color Reduction (Quantization)
        update_progress(70.0, "color_reduction", "Reducing colors to production palette...")
        quantized_img, palette = reduce_colors(current_img, config.color_reduction)
        pipeline_data["quantized"] = quantized_img
        pipeline_data["palette"] = palette

        # Step 8: Vectorization
        update_progress(80.0, "vectorization", "Vectorizing contours to SVG...")
        svg_content = vectorize_contours(current_img, config.vectorization)
        pipeline_data["svg_content"] = svg_content

        # Step 9: Repeat Generation
        update_progress(85.0, "repeat_generation", "Creating seamless repeat tiles...")
        repeat_tile = generate_repeat(quantized_img, repeat_info, config.repeat_generation)
        pipeline_data["repeat_tile"] = repeat_tile

        # Step 10: Color Variants (Colorways)
        update_progress(90.0, "color_variants", "Generating alternative colorways...")
        variants = generate_color_variants(quantized_img, palette, config.color_variants)
        pipeline_data["variants"] = variants

        # Step 11: Output Generation (BMP, TIFF, PSD, PNG, SVG)
        update_progress(95.0, "output_generation", "Writing output files (TIFF, BMP, PSD, PNG, SVG)...")
        saved_files = generate_outputs(file_id, filename, pipeline_data, config.output_generation)
        pipeline_data["saved_files"] = saved_files

        update_progress(100.0, "complete", "Pipeline successfully processed!")
        return {
            "status": "SUCCESS",
            "saved_files": saved_files
        }

pipeline_service = PipelineService()
