import os
import shutil
from pathlib import Path
from PIL import Image

# Force configuration env variables for testing offline
os.environ["STORAGE_TYPE"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = "./data/storage"
os.environ["ALLOW_MODEL_DOWNLOADS"] = "false"

from app.core.config import settings
from app.core.logger import logger
from app.models.request_models import ProcessConfig
from app.services.pipeline_service import pipeline_service
from app.utils.image_utils import create_synthetic_textile_image

def test_cad_preprocessing_engine():
    logger.info("Initializing standalone CAD Preprocessing Engine tests...")
    
    test_job_id = "test_cad_run"
    storage_dir = Path(settings.LOCAL_STORAGE_DIR)
    job_storage_dir = storage_dir / test_job_id
    
    if job_storage_dir.exists():
        logger.debug(f"Cleaning up old test run folder: {job_storage_dir}")
        shutil.rmtree(job_storage_dir)
        
    job_storage_dir.mkdir(parents=True, exist_ok=True)
    
    filename = "cad_design_sample.png"
    logger.info("Generating synthetic textile image for CAD testing...")
    synthetic_img = create_synthetic_textile_image(width=256, height=256)
    
    # Save synthetic image
    input_path = job_storage_dir / filename
    synthetic_img.save(input_path, format="PNG")
    logger.info(f"Saved initial design to {input_path}")
    
    # Configure pipeline settings
    config = ProcessConfig()
    config.cad_engine.enabled = True
    config.cad_engine.dpi = 600
    config.cad_engine.generate_six_versions = True
    
    # Disable downstream heavy operations to keep the test fast
    config.background_removal.enabled = False
    config.object_detection.enabled = False
    config.pattern_detection.enabled = False
    config.inpainting.enabled = False
    config.color_separation.num_colors = 2
    config.color_reduction.palette_size = 2
    config.vectorization.enabled = False
    config.repeat_generation.horizontal_tiles = 1
    config.repeat_generation.vertical_tiles = 1
    config.color_variants.enabled = False
    config.output_generation.formats = ["PNG"]  # Standard pipeline output

    def test_progress_callback(percent: float, step: str, msg: str):
        print(f"[PROGRESS] {percent:.1f}% | Step: {step:<20} | Status: {msg}")

    logger.info("Running pipeline with Textile CAD Preprocessing Engine enabled...")
    try:
        results = pipeline_service.run_pipeline(
            file_id=test_job_id,
            filename=filename,
            config=config,
            progress_callback=test_progress_callback
        )
        
        logger.info("Pipeline executed successfully!")
        
        saved_files = results.get("saved_files", [])
        print("\n" + "="*50)
        print(f"GENERATED OUTPUT FILES ({len(saved_files)} files):")
        print("="*50)
        for index, file in enumerate(saved_files, 1):
            file_size_kb = os.path.getsize(job_storage_dir / file) / 1024
            print(f"{index}. {file:<45} | Size: {file_size_kb:6.2f} KB")
        print("="*50 + "\n")
        
        # 6 independent versions
        expected_versions = [
            "version1_balanced_restoration",
            "version2_maximum_sharpness",
            "version3_print_optimized",
            "version4_vectorization_optimized",
            "version5_repeat_detection_optimized",
            "version6_texcelle_import_optimized"
        ]
        
        for version in expected_versions:
            png_name = f"cad_design_sample_{version}.png"
            bmp_name = f"cad_design_sample_{version}.bmp"
            
            # Assert file existence
            assert png_name in saved_files, f"Missing PNG for {version}"
            assert bmp_name in saved_files, f"Missing BMP for {version}"
            
            png_path = job_storage_dir / png_name
            bmp_path = job_storage_dir / bmp_name
            
            # Verify image properties
            with Image.open(png_path) as png_img:
                # Verify Lanczos upscaling (image dimensions should be 1024x1024, scaled from 256x256)
                w, h = png_img.size
                assert w == 1024 and h == 1024, f"Expected 1024x1024 for {version}, got {w}x{h}"
                # Verify DPI
                dpi = png_img.info.get("dpi")
                # Pillow might save DPI as a tuple of floats or ints
                assert dpi is not None, f"DPI is not set for PNG: {version}"
                assert round(dpi[0]) == 600 and round(dpi[1]) == 600, f"Expected 600 DPI, got {dpi} for PNG: {version}"

            with Image.open(bmp_path) as bmp_img:
                w, h = bmp_img.size
                assert w == 1024 and h == 1024, f"Expected 1024x1024 for {version}, got {w}x{h}"
                # Verify DPI
                dpi = bmp_img.info.get("dpi")
                assert dpi is not None, f"DPI is not set for BMP: {version}"
                assert round(dpi[0]) == 600 and round(dpi[1]) == 600, f"Expected 600 DPI, got {dpi} for BMP: {version}"
                
        logger.info("All 6 versions successfully generated in high quality, 600 DPI, lossless formats!")
        print("CAD PREPROCESSING ENGINE TEST PASSED.")
        
    except Exception as e:
        logger.exception(f"CAD test failed: {e}")
        print("CAD PREPROCESSING ENGINE TEST FAILED.")
        exit(1)

if __name__ == "__main__":
    test_cad_preprocessing_engine()
