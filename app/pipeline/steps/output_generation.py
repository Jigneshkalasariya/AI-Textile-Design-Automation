import io
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from app.models.request_models import OutputGenerationConfig
from app.services.storage_service import storage_service
from app.core.logger import logger

def generate_outputs(
    file_id: str,
    original_filename: str,
    pipeline_data: Dict[str, Any],
    config: OutputGenerationConfig
) -> List[str]:
    """
    Saves high-quality production ready textile files to storage.
    Supported formats:
        - PNG: Web-ready lossless format
        - BMP: Flat bitmap for engraving machines
        - TIFF: Professional print format (with LZW compression)
        - SVG: Scalable vector graphic representation
        - PSD: Photoshop document format
    Also saves color variants and layer separations.
    """
    logger.info("Running Step 11: Output Generation")
    
    # Get base filename without extension
    base_name = Path(original_filename).stem
    saved_files = []

    # Get images to save
    repeat_tile = pipeline_data.get("repeat_tile")
    if repeat_tile is None:
        repeat_tile = pipeline_data.get("original")

    # Save target formats for the repeat tile
    formats = [f.upper() for f in config.formats]
    
    for fmt in formats:
        try:
            # Determine extension
            ext = fmt.lower()
            if ext == "tiff":
                ext = "tif"
                
            filename = f"{base_name}_repeat.{ext}"
            buffer = io.BytesIO()
            
            # Format-specific compression and settings
            if fmt == "TIFF":
                # Save with LZW compression (Texcelle standard)
                repeat_tile.save(buffer, format="TIFF", compression="tiff_lzw")
            elif fmt == "PNG":
                repeat_tile.save(buffer, format="PNG", optimize=True)
            elif fmt == "BMP":
                # BMP doesn't support transparency, convert to RGB
                rgb_tile = repeat_tile.convert("RGB")
                rgb_tile.save(buffer, format="BMP")
            elif fmt == "PSD":
                # Save as PSD format using psd-tools
                from psd_tools import PSDImage
                tile_rgba = repeat_tile.convert("RGBA")
                psd = PSDImage.new(mode="RGBA", size=tile_rgba.size, depth=8)
                psd.create_pixel_layer(tile_rgba, name="Repeat Tile", top=0, left=0)
                psd.save(buffer)
            elif fmt == "SVG":
                # SVG is saved as text
                svg_str = pipeline_data.get("svg_content", "")
                buffer.write(svg_str.encode("utf-8"))
            else:
                repeat_tile.save(buffer, format=fmt)

            # Save to storage service
            storage_service.save_file(file_id, filename, buffer.getvalue())
            saved_files.append(filename)
            logger.debug(f"Generated output file: {filename}")

        except Exception as e:
            logger.error(f"Failed to generate output for format {fmt}: {e}")

    # Save CAD preprocessed versions if generated
    cad_versions = pipeline_data.get("cad_versions", {})
    if cad_versions:
        # Resolve target /output/ folder in the workspace root
        output_dir = Path(__file__).resolve().parents[3] / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map version keys to output folder short names
        output_mappings = {
            "version1": "v1_balanced.bmp",
            "version2": "v2_sharp.bmp",
            "version3": "v3_print.bmp",
            "version4": "v4_vector.bmp",
            "version5": "v5_repeat.bmp",
            "version6": "v6_texcelle.bmp",
            "sketch_bw": "sketch_bw.bmp"
        }
        
        for version_key, version_img in cad_versions.items():
            version_names = {
                "version1": "version1_balanced_restoration",
                "version2": "version2_maximum_sharpness",
                "version3": "version3_print_optimized",
                "version4": "version4_vectorization_optimized",
                "version5": "version5_repeat_detection_optimized",
                "version6": "version6_texcelle_import_optimized",
                "sketch_bw": "sketch_bw"
            }
            version_name = version_names.get(version_key, version_key)
            
            # Export as PNG and BMP to standard storage
            for ext in ["png", "bmp"]:
                try:
                    filename = f"{base_name}_{version_name}.{ext}"
                    buffer = io.BytesIO()
                    
                    if ext == "png":
                        version_img.convert("RGBA").save(buffer, format="PNG", optimize=True, dpi=(600, 600))
                    elif ext == "bmp":
                        version_img.convert("RGB").save(buffer, format="BMP", dpi=(600, 600))
                    
                    storage_service.save_file(file_id, filename, buffer.getvalue())
                    saved_files.append(filename)
                    logger.info(f"Generated CAD version output: {filename} at 600 DPI")
                except Exception as e:
                    logger.error(f"Failed to generate CAD output for {version_name} ({ext}): {e}")
            
            # Export to top-level /output/ folder as 600 DPI BMP
            if version_key in output_mappings:
                short_name = output_mappings[version_key]
                short_path = output_dir / short_name
                try:
                    version_img.convert("RGB").save(short_path, format="BMP", dpi=(600, 600))
                    logger.info(f"Saved exact prompt required output: {short_path} (600 DPI BMP)")
                except Exception as e:
                    logger.error(f"Failed to save short name version output {short_name}: {e}")

    # Save colorways if generated
    variants = pipeline_data.get("variants", {})
    for variant_name, var_img in variants.items():
        try:
            filename = f"{base_name}_{variant_name}.png"
            buffer = io.BytesIO()
            var_img.convert("RGBA").save(buffer, format="PNG")
            storage_service.save_file(file_id, filename, buffer.getvalue())
            saved_files.append(filename)
            logger.debug(f"Generated colorway file: {filename}")
        except Exception as e:
            logger.error(f"Failed to save variant {variant_name}: {e}")

    # Storage Optimization: Skip saving individual color layer PNGs to disk/cloud.
    # layers = pipeline_data.get("layers", [])
    # separated_colors = pipeline_data.get("separated_colors", [])
    # for idx, (layer, color) in enumerate(zip(layers, separated_colors)):
    #     try:
    #         hex_color = f"{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    #         filename = f"{base_name}_layer_{idx}_{hex_color}.png"
    #         buffer = io.BytesIO()
    #         layer.save(buffer, format="PNG")
    #         storage_service.save_file(file_id, filename, buffer.getvalue())
    #         saved_files.append(filename)
    #         logger.debug(f"Generated color separation layer: {filename}")
    #     except Exception as e:
    #         logger.error(f"Failed to save color layer {idx}: {e}")
    logger.info("Skipping output generation of individual color layers to optimize storage.")

    logger.info(f"Output generation completed. Total files written: {len(saved_files)}")
    return saved_files
