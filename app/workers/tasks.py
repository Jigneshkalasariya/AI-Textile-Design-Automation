from typing import Dict, Any
import httpx
import os
import shutil
import urllib.parse
from datetime import datetime
from app.workers.celery_worker import celery_app
from app.models.request_models import ProcessConfig
from app.services.pipeline_service import pipeline_service
from app.services.storage_service import storage_service
from app.services.cloudinary_service import upload_image
from app.core.database import SessionLocal
from app.models.db_models import get_model
from app.core.logger import logger

@celery_app.task(bind=True, name="app.workers.tasks.process_textile_image")
def process_textile_image(self, file_id: str, filename: str, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous Celery task that executes the 11-step pipeline.
    Updates the task metadata periodically to track job progress.
    """
    logger.info(f"Worker received processing request for job ID {file_id}")
    
    # 1. Parse configuration dict back to Pydantic object
    try:
        config = ProcessConfig.model_validate(config_dict)
    except Exception as e:
        logger.error(f"Task configuration validation failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": f"Invalid configuration: {str(e)}"}
        )
        raise ValueError(f"Invalid config dictionary: {e}")

    # 2. Progress callback function
    def progress_callback(progress_percent: float, current_step: str, status_msg: str):
        # Update celery state custom metadata
        self.update_state(
            state="PROCESSING",
            meta={
                "progress": progress_percent,
                "step": current_step,
                "message": status_msg
            }
        )
        try:
            with SessionLocal() as progress_db:
                FileAsset = get_model('file_assets')
                asset = progress_db.query(FileAsset).filter(FileAsset.id == file_id).first()
                if asset and hasattr(asset, 'processing_percentage'):
                    asset.processing_percentage = progress_percent
                    progress_db.commit()
        except Exception as e:
            logger.error(f"Failed to update progress in DB: {e}")

    # 3. Execute pipeline
    try:
        # Run pipeline
        result = pipeline_service.run_pipeline(
            file_id=file_id,
            filename=filename,
            config=config,
            progress_callback=progress_callback
        )
        return result
    except Exception as e:
        logger.exception(f"Pipeline crashed during background task execution: {e}")
        # Re-raise to trigger Celery's standard failure handler
        raise e

@celery_app.task(name="app.workers.tasks.check_queue_task")
def check_queue_task():
    """
    Periodic task to check file_assets table for QUEUE status and trigger processing.
    """
    logger.info("Checking file_assets for QUEUE items...")
    db = SessionLocal()
    try:
        FileAsset = get_model('file_assets')
        # Find queued assets
        queued_assets = db.query(FileAsset).filter(FileAsset.status == 'QUEUE').all()
        for asset in queued_assets:
            logger.info(f"Found QUEUE asset: {asset.id} - URL: {asset.enhanced_cloudinary_url}")
            asset.status = 'PROCESSING'
            db.commit()
            # Trigger celery task for individual asset
            process_queued_asset.delay(str(asset.id), asset.enhanced_cloudinary_url)
    except Exception as e:
        logger.error(f"Error checking queue: {e}")
        db.rollback()
    finally:
        db.close()

@celery_app.task(bind=True, name="app.workers.tasks.process_queued_asset")
def process_queued_asset(self, asset_id: str, url: str):
    """
    Downloads asset, runs pipeline, uploads to Cloudinary, and saves to similar_files table.
    """
    logger.info(f"Processing queued asset {asset_id} from {url}")
    db = SessionLocal()
    
    # Extract base folder for Cloudinary from original url
    target_cloudinary_folder = f"similar_files/{asset_id}"
    try:
        parsed = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(parsed.path)
        if "/upload/" in path:
            post_upload = path.split("/upload/")[1]
            parts = post_upload.split("/")
            if parts[0].startswith("v") and parts[0][1:].isdigit():
                parts = parts[1:]
            if "." in parts[-1]:
                parts[-1] = parts[-1].rsplit(".", 1)[0]
            target_cloudinary_folder = f"{'/'.join(parts)}/similar_files"
    except Exception as e:
        logger.warning(f"Could not parse original URL for folder structure: {e}")
        
    try:
        # Download the original image
        filename = url.split("/")[-1]
        if not filename or "?" in filename:
            filename = "source_image.jpg"
            
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
            image_data = response.content
            
        # Save locally so pipeline can read it
        storage_service.save_file(asset_id, filename, image_data)
        
        # Run pipeline with automated configuration
        # cad_only=False ensures full 11-step execution and saving of all outputs
        config = ProcessConfig(
            cad_only=True,
            cad_engine={
                "enabled": True,
                "prompt": (
                    "TEXTILE CAD AND TEXCELLE IMAGE PREPROCESSING ENGINE\n\n"
                    "Process the uploaded textile artwork into production-ready Texcelle files without changing its design identity.\n\n"
                    "Every output must have exactly the same pixel width, pixel height, aspect ratio, orientation, crop, and canvas boundaries as the uploaded image. Do not upscale, downscale, crop, extend, tile, rotate, or recompose the artwork.\n\n"
                    "Preserve every motif, flower, leaf, ornament, border, curve, outline, internal line, relative position, spacing, proportion, symmetry, repeat boundary, edge continuity, legitimate background, fine detail, texture, original color identity, and palette relationship.\n\n"
                    "Never add, invent, remove, regenerate, redesign, stylize, move, duplicate, or hallucinate design content. Never change motif geometry, repeat structure, or legitimate design backgrounds. Do not add grids, labels, watermarks, shadows, frames, or text.\n\n"
                    "Use the original uploaded image as the independent source for every output. Apply conservative edge-preserving cleaning only to defects that are clearly not artwork. Preserve geometry and canvas dimensions. Restore faded colors with LAB luminance correction while preserving hues and avoiding clipping. Apply controlled sharpening without halos.\n\n"
                    "Generate exactly four variants:\n"
                    "1. master_enhanced: naturally restored original palette and texture with balanced contrast and controlled sharpness.\n"
                    "2. sketch_bw: CAD line drawing containing only black 0 and white 255, with no grayscale, transparency, shading, hatching, gradients, or texture.\n"
                    "3. color_variant_soft: subtle contrast and color restoration without recoloring.\n"
                    "4. color_variant_vibrant: moderate contrast and saturation while preserving original color identities and avoiding clipping.\n\n"
                    "Export each variant independently as uncompressed BMP and lossless PNG at exactly the original pixel dimensions with 600 x 600 DPI metadata and no transparency."
                ),
                "dpi": 600,
                "generate_six_versions": False
            },
            enhance={
                "denoise_strength": 3.0,
                "sharpen_strength": 1.5,
                "use_realesrgan": False
            },
            background_removal={
                "enabled": True,
                "model_name": "sam2",
                "use_grounding_dino": False,
                "alpha_matting": False
            },
            object_detection={
                "enabled": True,
                "model": "yolo11",
                "confidence": 0.25
            },
            pattern_detection={
                "enabled": True,
                "search_grid_size": 64
            },
            inpainting={
                "enabled": False,
                "model": "flux",
                "prompt": "seamless matching textile pattern fabric",
                "negative_prompt": "seams, lines, bad quality, blurry",
                "strength": 0.8
            },
            color_separation={
                "num_colors": 8,
                "color_space": "LAB"
            },
            color_reduction={
                "palette_size": 8,
                "dither": True
            },
            vectorization={
                "enabled": True,
                "method": "potrace",
                "simplify_tolerance": 1.0
            },
            repeat_generation={
                "repeat_type": "straight",
                "horizontal_tiles": 3,
                "vertical_tiles": 3
            },
            color_variants={
                "enabled": True,
                "variant_count": 3,
                "harmony_types": ["complementary", "analogous", "triad"]
            },
            output_generation={
                "formats": ["PNG", "SVG"]
            }
        )
        
        def progress_callback(progress_percent: float, current_step: str, status_msg: str):
            self.update_state(
                state="PROCESSING",
                meta={
                    "progress": progress_percent,
                    "step": current_step,
                    "message": status_msg
                }
            )
            try:
                with SessionLocal() as progress_db:
                    FileAsset = get_model('file_assets')
                    asset = progress_db.query(FileAsset).filter(FileAsset.id == asset_id).first()
                    if asset and hasattr(asset, 'processing_percentage'):
                        asset.processing_percentage = progress_percent
                        progress_db.commit()
            except Exception as e:
                logger.error(f"Failed to update progress in DB: {e}")
            
        pipeline_service.run_pipeline(
            file_id=asset_id,
            filename=filename,
            config=config,
            progress_callback=progress_callback
        )
        
        # List all generated files
        generated_files = storage_service.list_files(asset_id)
        
        SimilarFile = get_model('similar_files')
        FileAsset = get_model('file_assets')
        
        # Upload all generated outputs to Cloudinary and insert into similar_files
        for gen_filename in generated_files:
            if gen_filename == filename or gen_filename == "job_status.json":
                continue # Skip original and status file
                
            local_path = storage_service.get_local_path(asset_id, gen_filename)
            
            # Convert to JPG if it's a huge BMP to avoid Cloudinary 10MB limit
            try:
                from PIL import Image
                import os
                file_size = os.path.getsize(local_path)
                # If file > 8MB or it's a BMP, convert to JPG
                if file_size > 8 * 1024 * 1024 or str(local_path).lower().endswith(".bmp"):
                    logger.info(f"Converting {gen_filename} to JPG to save space...")
                    img = Image.open(local_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    new_local_path = str(local_path).rsplit(".", 1)[0] + ".jpg"
                    img.save(new_local_path, "JPEG", quality=85)
                    local_path = new_local_path
                    gen_filename = gen_filename.rsplit(".", 1)[0] + ".jpg"
            except Exception as e:
                logger.warning(f"Failed to convert image {gen_filename}: {e}")

            logger.info(f"Uploading generated file {gen_filename} to Cloudinary...")
            
            upload_response = upload_image(str(local_path), folder=target_cloudinary_folder)
            cloud_url = upload_response.get("secure_url")
            cloud_id = upload_response.get("public_id")
            
            if cloud_url:
                # Save to similar_files table
                # Passing kwargs for new columns. Ensure they are added in NestJS side!
                similar_kwargs = {
                    "file_asset_id": asset_id,
                    "url": cloud_url,
                    "created_at": datetime.utcnow()
                }
                
                # Check if columns exist in reflected model before adding to avoid crash if not migrated yet
                if hasattr(SimilarFile, 'cloudinary_id'):
                    similar_kwargs['cloudinary_id'] = cloud_id
                if hasattr(SimilarFile, 'filename'):
                    similar_kwargs['filename'] = gen_filename
                    
                new_similar = SimilarFile(**similar_kwargs)
                db.add(new_similar)
                
        # Update original asset status to Completed
        asset = db.query(FileAsset).filter(FileAsset.id == asset_id).first()
        if asset:
            asset.status = 'COMPLETED'
            if hasattr(asset, 'processing_percentage'):
                asset.processing_percentage = 100
            
        db.commit()
        logger.info(f"Successfully processed queued asset {asset_id}")
        
        # Cleanup local directory
        local_job_dir = storage_service.local_dir / asset_id
        if local_job_dir.exists():
            shutil.rmtree(local_job_dir, ignore_errors=True)
            
    except Exception as e:
        logger.exception(f"Failed to process queued asset {asset_id}: {e}")
        db.rollback()
        
        # Mark as failed in DB
        try:
            FileAsset = get_model('file_assets')
            asset = db.query(FileAsset).filter(FileAsset.id == asset_id).first()
            if asset:
                asset.status = 'FAILED'
                db.commit()
        except:
            pass
            
        raise e
    finally:
        db.close()
