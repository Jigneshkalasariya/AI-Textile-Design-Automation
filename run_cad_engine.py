import os
import json
from app.services.cad_textile_engine import cad_engine_processor
from app.utils.image_utils import create_synthetic_textile_image

def main():
    # Ensure test synthetic image exists
    input_path = "./data/cad_sample_input.png"
    os.makedirs("./data", exist_ok=True)
    
    synthetic_img = create_synthetic_textile_image(width=512, height=512)
    synthetic_img.save(input_path, format="PNG")
    
    # Process image with Textile CAD AI Engine
    result = cad_engine_processor.process_image(input_image=input_path, output_dir="./output")
    
    # Print formatted JSON output
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
