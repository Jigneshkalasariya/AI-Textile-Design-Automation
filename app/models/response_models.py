from typing import List, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    file_id: str = Field(..., description="Unique identifier generated for the uploaded file")
    filename: str = Field(..., description="Sanitized name of the uploaded file")
    storage_type: str = Field(..., description="Configured storage backend used (e.g. local, s3)")
    message: str = Field(..., description="Status message indicating upload success or details")

class JobSubmitResponse(BaseModel):
    job_id: str = Field(..., description="Unique celery task / job identifier")
    status: str = Field(..., description="Initial celery task status (e.g. PENDING)")
    message: str = Field(..., description="Submission status message")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status of the job (e.g. PENDING, SUCCESS, FAILURE, PROCESSING)")
    progress: float = Field(default=0.0, description="Progress percentage (0.0 to 100.0)")
    current_step: Optional[str] = Field(default=None, description="Active processing step name")
    message: Optional[str] = Field(default=None, description="Status description or step details")
    output_files: List[str] = Field(default_factory=list, description="List of generated output file names (if completed)")
    error: Optional[str] = Field(default=None, description="Failure details if job status is FAILURE")
