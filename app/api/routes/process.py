from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Body
from app.models.request_models import ProcessConfig
from app.models.response_models import JobSubmitResponse
from app.services.storage_service import storage_service
from app.workers.tasks import process_textile_image
from app.utils.file_utils import is_allowed_file
from app.services.queue_service import queue_service
from app.core.config import settings
from app.core.logger import logger
from typing import Dict, Any, Optional

router = APIRouter(prefix="/api")

def run_local_pipeline_job(job_id: str, filename: str, config_dict: Dict[str, Any]):
    logger.info(f"Starting local background processing for job {job_id}")
    queue_service.update_local_job_status(
        job_id=job_id,
        status="PROCESSING",
        progress=0.0,
        current_step="initializing",
        message="Starting pipeline processing offline..."
    )
    
    try:
        from app.services.pipeline_service import pipeline_service
        config = ProcessConfig.model_validate(config_dict)
        
        def progress_callback(progress_percent: float, current_step: str, status_msg: str):
            queue_service.update_local_job_status(
                job_id=job_id,
                status="PROCESSING",
                progress=progress_percent,
                current_step=current_step,
                message=status_msg
            )
            
        pipeline_service.run_pipeline(
            file_id=job_id,
            filename=filename,
            config=config,
            progress_callback=progress_callback
        )
        
        queue_service.update_local_job_status(
            job_id=job_id,
            status="SUCCESS",
            progress=100.0,
            current_step="complete",
            message="Pipeline completed successfully."
        )
        logger.info(f"Local background processing completed successfully for job {job_id}")
    except Exception as e:
        logger.exception(f"Local background processing failed for job {job_id}: {e}")
        queue_service.update_local_job_status(
            job_id=job_id,
            status="FAILURE",
            progress=100.0,
            current_step="error",
            message="Pipeline execution failed.",
            error=str(e)
        )

@router.post(
    "/process/{file_id}",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Process an uploaded textile image",
    description=(
        "Uses the original image already stored under file_id. By default, generates "
        "the eight dimension-preserving Textile CAD/Texcelle BMP and PNG outputs."
    ),
)
async def process_image(
    file_id: str,
    background_tasks: BackgroundTasks,
    config: Optional[ProcessConfig] = Body(default=None),
) -> JobSubmitResponse:
    config = config or ProcessConfig()

    # 1. Locate original file
    files = storage_service.list_files(file_id)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded files found for ID {file_id}"
        )
    
    # Filter generated artifacts so rerunning a file_id always starts from the
    # original uploaded source rather than from a previous generated variant.
    generated_names = {
        "master_enhanced.bmp",
        "master_enhanced.png",
        "sketch_bw.bmp",
        "sketch_bw.png",
        "color_variant_soft.bmp",
        "color_variant_soft.png",
        "color_variant_vibrant.bmp",
        "color_variant_vibrant.png",
    }
    generated_markers = (
        "_repeat",
        "_layer_",
        "colorway_",
        "_master_enhanced",
        "_sketch_bw",
        "_color_variant_",
        "_version",
    )
    original_file = None
    for f in files:
        lower_name = f.lower()
        if (
            is_allowed_file(f)
            and lower_name not in generated_names
            and not any(marker in lower_name for marker in generated_markers)
        ):
            original_file = f
            break
            
    if not original_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original source image file not found in storage."
        )

    try:
        # Check if we should use Celery or local fallback
        use_celery = settings.CELERY_ENABLED and queue_service.is_redis_available()
        
        if use_celery:
            logger.info(f"Triggering Celery task for file {original_file} under ID {file_id}")
            # 2. Dispatch task asynchronously using celery
            # We can map the Celery task ID to be the job_id
            task = process_textile_image.apply_async(
                args=[file_id, original_file, config.model_dump()],
                task_id=file_id # Force Celery task ID to match file_id for 1:1 mapping simplicity
            )
            
            return JobSubmitResponse(
                job_id=task.id,
                status=task.status,
                message="Processing job triggered successfully (via Celery)"
            )
        else:
            logger.info(f"Queueing local background task for job {file_id} (offline fallback mode)")
            
            # Initialize job status file with PENDING
            queue_service.update_local_job_status(
                job_id=file_id,
                status="PENDING",
                progress=0.0,
                current_step="pending",
                message="Job is queued locally."
            )
            
            # Dispatch local background task
            background_tasks.add_task(
                run_local_pipeline_job,
                file_id,
                original_file,
                config.model_dump()
            )
            
            return JobSubmitResponse(
                job_id=file_id,
                status="PENDING",
                message="Processing job triggered successfully (via local fallback)"
            )
        
    except Exception as e:
        logger.error(f"Failed to submit processing job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue processing job: {str(e)}"
        )
