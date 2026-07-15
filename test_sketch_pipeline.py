import os
import time
import shutil
from pathlib import Path
import cv2
import numpy as np

# Force environment configuration
os.environ["STORAGE_TYPE"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = "./data/storage"

from app.core.logger import logger
from app.utils.image_utils import create_synthetic_textile_image
from app.services.sketch_service import process_image, HAS_TORCH

def run_tests():
    logger.info("Initializing standalone Textile Sketch Processing Engine tests...")
    
    # 1. Setup clean directory structure for testing
    test_dir = Path("./data/sketch_test_runs")
    if test_dir.exists():
        logger.debug(f"Cleaning up old test run folder: {test_dir}")
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate a synthetic textile image representing background, motifs, patterns
    logger.info("Generating synthetic textile design image...")
    synthetic_pil = create_synthetic_textile_image(width=512, height=512)
    input_path = test_dir / "textile_design_sample.png"
    synthetic_pil.save(input_path, format="PNG")
    logger.info(f"Saved initial design to {input_path}")
    
    input_path_str = str(input_path.absolute())
    output_dir_str = str(test_dir.absolute())
    
    test_runs = [
        {
            "name": "Default Sketch Pipeline (BMP, Grid ON, AI OFF, Pencil)",
            "format": "bmp",
            "use_ai": False,
            "grid": True,
            "kwargs": {
                "sketch_method": "pencil"
            }
        },
        {
            "name": "AI-Enhanced Sketch (PNG, Grid OFF, AI ON, Pencil, Erode Line Thicken)",
            "format": "png",
            "use_ai": True,
            "grid": False,
            "kwargs": {
                "sketch_method": "pencil",
                "morphology": "erode",
                "morphology_kernel": 3
            }
        },
        {
            "name": "Threshold Sketch (JPG, Grid OFF, AI OFF, Adaptive Threshold, Dilate Line Thin)",
            "format": "jpg",
            "use_ai": False,
            "grid": False,
            "kwargs": {
                "sketch_method": "threshold",
                "threshold_block_size": 11,
                "threshold_c": 2,
                "morphology": "dilate",
                "morphology_kernel": 3
            }
        }
    ]
    
    results = []
    
    for run in test_runs:
        print("\n" + "="*80)
        print(f"TEST CASE: {run['name']}")
        print("="*80)
        
        start_time = time.time()
        try:
            out_path = process_image(
                input_path=input_path_str,
                output_dir=output_dir_str,
                format=run["format"],
                use_ai=run["use_ai"],
                grid=run["grid"],
                **run["kwargs"]
            )
            elapsed = time.time() - start_time
            print(f"Success! Output file: {out_path}")
            print(f"Time Taken: {elapsed:.3f} seconds")
            
            # Assert file exists
            assert os.path.exists(out_path), "Output file does not exist!"
            # Assert file is not empty
            file_size = os.path.getsize(out_path)
            assert file_size > 0, "Output file is empty!"
            
            # Try reading the image back with OpenCV to check integrity
            loaded_img = cv2.imread(out_path)
            assert loaded_img is not None, "Failed to decode output image with OpenCV!"
            h, w = loaded_img.shape[:2]
            assert h == 512 and w == 512, f"Output size mismatch: expected 512x512, got {w}x{h}"
            
            print(f"File verified. Dimensions: {w}x{h}, File Size: {file_size / 1024:.2f} KB")
            
            # Performance check
            assert elapsed < 2.0, f"Processing time exceeded 2.0 seconds! Took {elapsed:.3f}s"
            
            results.append({
                "name": run["name"],
                "status": "PASSED",
                "output_file": out_path,
                "size_kb": file_size / 1024.0,
                "time_sec": elapsed
            })
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Test case failed: {run['name']}")
            print(f"Failure! Error: {e}")
            results.append({
                "name": run["name"],
                "status": f"FAILED ({str(e)})",
                "output_file": "",
                "size_kb": 0.0,
                "time_sec": elapsed
            })
            
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY:")
    print("="*80)
    all_passed = True
    for res in results:
        print(f"- {res['name']}: {res['status']}")
        if res['status'] == "PASSED":
            print(f"  File: {res['output_file']}")
            print(f"  Size: {res['size_kb']:.2f} KB | Time: {res['time_sec']:.3f}s")
        else:
            all_passed = False
    print("="*80)
    
    if all_passed:
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("\nSOME TESTS FAILED.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
