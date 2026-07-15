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
from app.services.storage_service import storage_service

def test_complete_pipeline():
    logger.info("Initializing standalone pipeline testing script...")
    
    # 1. Setup clean directory structure for testing
    test_job_id = "test_pipeline_run"
    storage_dir = Path(settings.LOCAL_STORAGE_DIR)
    job_storage_dir = storage_dir / test_job_id
    
    if job_storage_dir.exists():
        logger.debug(f"Cleaning up old test run folder: {job_storage_dir}")
        shutil.rmtree(job_storage_dir)
        
    job_storage_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate a synthetic textile image representing background, motifs, patterns
    filename = "textile_design_sample.png"
    logger.info(f"Generating synthetic textile design image: {filename}")
    synthetic_img = create_synthetic_textile_image(width=512, height=512)
    
    # 3. Save synthetic image into the job's input storage location
    input_path = job_storage_dir / filename
    synthetic_img.save(input_path, format="PNG")
    logger.info(f"Saved initial design to {input_path}")
    
    # 4. Configure pipeline settings
    # We use default settings (which run OpenCV/Pillow fallbacks for ML models offline)
    config = ProcessConfig()
    
    # Explicitly enable colorways variant generation and vectorization
    config.color_variants.enabled = True
    config.color_variants.variant_count = 3
    config.vectorization.enabled = True
    
    # 5. Define console progress callback
    def test_progress_callback(percent: float, step: str, msg: str):
        print(f"[PROGRESS] {percent:.1f}% | Step: {step:<20} | Status: {msg}")

    # 6. Execute pipeline
    logger.info("Triggering 11-step pipeline execution...")
    try:
        results = pipeline_service.run_pipeline(
            file_id=test_job_id,
            filename=filename,
            config=config,
            progress_callback=test_progress_callback
        )
        
        logger.info("Pipeline executed successfully without crashes!")
        
        # 7. Print list of generated files
        saved_files = results.get("saved_files", [])
        print("\n" + "="*50)
        print(f"GENERATED OUTPUT FILES ({len(saved_files)} files):")
        print("="*50)
        for index, file in enumerate(saved_files, 1):
            file_size_kb = os.path.getsize(job_storage_dir / file) / 1024
            print(f"{index}. {file:<45} | Size: {file_size_kb:6.2f} KB")
        print("="*50 + "\n")
        
        # 8. Assertions / Checks
        # Standard repeat outputs must exist
        assert any("_repeat.png" in f for f in saved_files), "Missing PNG repeat tile"
        assert any("_repeat.tif" in f for f in saved_files), "Missing TIFF repeat tile"
        assert any("_repeat.bmp" in f for f in saved_files), "Missing BMP repeat tile"
        assert any("_repeat.svg" in f for f in saved_files), "Missing SVG repeat tile"
        assert any("_repeat.psd" in f for f in saved_files), "Missing PSD repeat tile"
        assert any("colorway_complementary" in f for f in saved_files), "Missing complementary colorway"
        assert any("colorway_analogous" in f for f in saved_files), "Missing analogous colorway"
        assert any("_layer_" in f for f in saved_files), "Missing KMeans color separation layers"
        
        logger.info("All generated files verified successfully!")
        print("TEST PASSED.")
        
    except Exception as e:
        logger.exception(f"Pipeline test failed with error: {e}")
        print("TEST FAILED.")
        exit(1)

if __name__ == "__main__":
    test_complete_pipeline()
