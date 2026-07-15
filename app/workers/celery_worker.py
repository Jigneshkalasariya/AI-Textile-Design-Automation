from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "textile_pipeline_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

# Optional configurations
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Clean up results after 2 days
    result_expires=172800,
)
