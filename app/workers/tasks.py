from typing import Dict, Any
from app.workers.celery_worker import celery_app
from app.models.request_models import ProcessConfig
from app.services.pipeline_service import pipeline_service
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
