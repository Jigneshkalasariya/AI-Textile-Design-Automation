from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "textile_pipeline_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

from celery.signals import worker_process_init
from app.core.database import init_db
from app.services.cloudinary_service import init_cloudinary

@worker_process_init.connect
def init_celery_services(**kwargs):
    init_db()
    init_cloudinary()

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

# Configure periodic tasks (Celery Beat)
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "check-queued-assets-every-minute": {
        "task": "app.workers.tasks.check_queue_task",
        "schedule": crontab(minute="*"),  # Every minute
    },
}
