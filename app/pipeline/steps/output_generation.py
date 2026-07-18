import io
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from app.models.request_models import OutputGenerationConfig
from app.services.storage_service import storage_service
from app.core.logger import logger

REQUIRED_CAD_OUTPUTS = (
    "master_enhanced",
    "sketch_bw",
    "color_variant_soft",
    "color_variant_vibrant",
)


def _encode_cad_output(
    image: Image.Image,
    output_format: str,
    is_sketch: bool,
    dpi: int,
) -> bytes:
    buffer = io.BytesIO()
    if is_sketch:
        export_image = image.convert("L").point(
            lambda value: 255 if value >= 128 else 0,
            mode="1" if output_format == "BMP" else "L",
        )
    else:
        export_image = image.convert("RGB")

    save_options = {"format": output_format, "dpi": (dpi, dpi)}
    if output_format == "PNG":
        save_options.update({"optimize": True, "compress_level": 9})
    export_image.save(buffer, **save_options)
    return buffer.getvalue()


def _validate_cad_output(
    content: bytes,
    filename: str,
    expected_size: tuple[int, int],
    is_sketch: bool,
    dpi: int,
) -> None:
    if filename.endswith(".bmp"):
        if content[:2] != b"BM":
            raise ValueError(f"{filename} is not a valid BMP")
        compression = int.from_bytes(content[30:34], byteorder="little")
        if compression != 0:
            raise ValueError(f"{filename} BMP compression must be BI_RGB (0)")

    with Image.open(io.BytesIO(content)) as reopened:
        reopened.load()
        if reopened.size != expected_size:
            raise ValueError(
                f"{filename} dimensions {reopened.size} do not match source {expected_size}"
            )
        exported_dpi = reopened.info.get("dpi")
        if not exported_dpi or any(round(value) != dpi for value in exported_dpi[:2]):
            raise ValueError(f"{filename} does not contain {dpi} x {dpi} DPI metadata")
        if reopened.mode in {"RGBA", "LA"} or "transparency" in reopened.info:
            raise ValueError(f"{filename} contains transparency")
        if is_sketch:
            histogram = reopened.convert("L").histogram()
            if any(histogram[1:255]):
                raise ValueError(f"{filename} contains non-binary sketch pixels")


def generate_cad_outputs(
    file_id: str,
    cad_versions: Dict[str, Image.Image],
    source_size: tuple[int, int],
    dpi: int = 600,
) -> List[str]:
    """Write and validate exactly the eight production Texcelle files."""
    missing = [name for name in REQUIRED_CAD_OUTPUTS if name not in cad_versions]
    if missing:
        raise ValueError(f"Missing required CAD variants: {', '.join(missing)}")

    output_dir = Path(__file__).resolve().parents[3] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []

    for variant_name in REQUIRED_CAD_OUTPUTS:
        image = cad_versions[variant_name]
        if image.size != source_size:
            raise ValueError(
                f"{variant_name} dimensions {image.size} do not match source {source_size}"
            )

        for extension, output_format in (("bmp", "BMP"), ("png", "PNG")):
            filename = f"{variant_name}.{extension}"
            content = _encode_cad_output(
                image=image,
                output_format=output_format,
                is_sketch=variant_name == "sketch_bw",
                dpi=dpi,
            )
            _validate_cad_output(
                content=content,
                filename=filename,
                expected_size=source_size,
                is_sketch=variant_name == "sketch_bw",
                dpi=dpi,
            )
            storage_service.save_file(file_id, filename, content)
            (output_dir / filename).write_bytes(content)
            saved_files.append(filename)
            logger.info(f"Generated and validated CAD output: {filename}")

    if len(saved_files) != 8:
        raise ValueError(f"Expected exactly 8 CAD outputs, generated {len(saved_files)}")
    return saved_files


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
        
        # Map version keys to output folder short names (legacy)
        output_mappings = {
            "version1": "v1_balanced.bmp",
            "version2": "v2_sharp.bmp",
            "version3": "v3_print.bmp",
            "version4": "v4_vector.bmp",
            "version5": "v5_repeat.bmp",
            "version6": "v6_texcelle.bmp",
            "sketch_bw": "sketch_bw.bmp"
        }
        
        # Define the exact names of versions to save in the job storage
        version_names = {
            "version1": "version1_balanced_restoration",
            "version2": "version2_maximum_sharpness",
            "version3": "version3_print_optimized",
            "version4": "version4_vectorization_optimized",
            "version5": "version5_repeat_detection_optimized",
            "version6": "version6_texcelle_import_optimized",
            "sketch_bw": "sketch_bw",
            "master_enhanced": "master_enhanced",
            "color_variant_soft": "color_variant_soft",
            "color_variant_vibrant": "color_variant_vibrant"
        }
        
        for version_key, version_img in cad_versions.items():
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
                    logger.info(f"Saved legacy CAD output: {short_path} (600 DPI BMP)")
                except Exception as e:
                    logger.error(f"Failed to save legacy CAD output {short_name}: {e}")

        # Export the 4 REQUIRED outputs to the top-level /output/ folder in both BMP and PNG formats
        required_outputs = {
            "master_enhanced": "master_enhanced",
            "sketch_bw": "sketch_bw",
            "color_variant_soft": "color_variant_soft",
            "color_variant_vibrant": "color_variant_vibrant"
        }

        for req_key, req_name in required_outputs.items():
            if req_key in cad_versions:
                version_img = cad_versions[req_key]
                
                # 1. Export as BMP to /output/
                bmp_path = output_dir / f"{req_name}.bmp"
                try:
                    version_img.convert("RGB").save(bmp_path, format="BMP", dpi=(600, 600))
                    logger.info(f"Saved required CAD output: {bmp_path} (600 DPI BMP)")
                except Exception as e:
                    logger.error(f"Failed to save required BMP output {req_name}: {e}")
                
                # 2. Export as PNG to /output/
                png_path = output_dir / f"{req_name}.png"
                try:
                    version_img.convert("RGBA").save(png_path, format="PNG", optimize=True, dpi=(600, 600))
                    logger.info(f"Saved required CAD output: {png_path} (600 DPI PNG)")
                except Exception as e:
                    logger.error(f"Failed to save required PNG output {req_name}: {e}")

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
