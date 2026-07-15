import json
import time
import redis
from pathlib import Path
from typing import Dict, Any, Optional
from celery.result import AsyncResult
from app.workers.celery_worker import celery_app
from app.models.response_models import JobStatusResponse
from app.services.storage_service import storage_service
from app.core.config import settings
from app.core.logger import logger

class QueueService:
    _redis_checked_at = 0.0
    _redis_available = False

    @classmethod
    def is_redis_available(cls) -> bool:
        """
        Checks if Redis broker is reachable. Caches the result for 5 seconds
        to prevent blocking requests when Redis is down.
        """
        now = time.time()
        if now - cls._redis_checked_at < 5.0:
            return cls._redis_available
            
        cls._redis_checked_at = now
        try:
            r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.5)
            r.ping()
            cls._redis_available = True
        except Exception:
            cls._redis_available = False
            logger.warning(f"Redis is unreachable at {settings.REDIS_URL}. Falling back to local offline mode.")
        return cls._redis_available

    @staticmethod
    def get_local_status_path(job_id: str) -> Path:
        return settings.local_storage_path / job_id / "job_status.json"

    @classmethod
    def get_local_job_status(cls, job_id: str) -> Optional[Dict[str, Any]]:
        status_file = cls.get_local_status_path(job_id)
        if not status_file.exists():
            return None
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read local status file for {job_id}: {e}")
            return None

    @classmethod
    def update_local_job_status(
        cls,
        job_id: str,
        status: str,
        progress: float = 0.0,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        status_file = cls.get_local_status_path(job_id)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        
        status_data = {
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "message": message,
            "error": error
        }
        try:
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write local status file for {job_id}: {e}")

    @classmethod
    def get_job_status(cls, job_id: str) -> JobStatusResponse:
        """
        Retrieves the state and metadata of a background job.
        If Celery is disabled or Redis is unreachable, falls back to reading
        from local status file.
        """
        status = "PENDING"
        progress = 0.0
        current_step = None
        message = None
        error = None
        output_files = []

        if settings.CELERY_ENABLED and cls.is_redis_available():
            try:
                res = AsyncResult(job_id, app=celery_app)
                status = res.status
                
                if status == "SUCCESS":
                    progress = 100.0
                    current_step = "complete"
                    message = "Pipeline completed successfully."
                    output_files = [f for f in storage_service.list_files(job_id) if f != "job_status.json"]
                elif status == "FAILURE":
                    progress = 100.0
                    error = str(res.result)
                    message = "Pipeline execution failed."
                elif status == "PENDING":
                    # Check if there is a local fallback job status instead
                    local_meta = cls.get_local_job_status(job_id)
                    if local_meta:
                        status = local_meta.get("status", "PENDING")
                        progress = local_meta.get("progress", 0.0)
                        current_step = local_meta.get("current_step")
                        message = local_meta.get("message")
                        error = local_meta.get("error")
                        output_files = [f for f in storage_service.list_files(job_id) if f != "job_status.json"]
                    else:
                        progress = 0.0
                        message = "Job is waiting in queue."
                else:
                    meta = res.info
                    if isinstance(meta, dict):
                        progress = meta.get("progress", 0.0)
                        current_step = meta.get("step", "processing")
                        message = meta.get("message", "Processing image pipeline...")
                    else:
                        progress = 10.0
                        message = "Job execution started."
                        
                return JobStatusResponse(
                    job_id=job_id,
                    status=status,
                    progress=progress,
                    current_step=current_step,
                    message=message,
                    output_files=output_files,
                    error=error
                )
            except Exception as e:
                logger.error(f"Error querying Celery for {job_id}: {e}. Falling back to local state check.")

        # Fallback for when Celery/Redis is not used or unreachable
        local_meta = cls.get_local_job_status(job_id)
        if local_meta:
            status = local_meta.get("status", "PENDING")
            progress = local_meta.get("progress", 0.0)
            current_step = local_meta.get("current_step")
            message = local_meta.get("message")
            error = local_meta.get("error")
            output_files = [f for f in storage_service.list_files(job_id) if f != "job_status.json"]
        else:
            status = "PENDING"
            message = "Job is waiting or initializing."

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress,
            current_step=current_step,
            message=message,
            output_files=output_files,
            error=error
        )

queue_service = QueueService()
