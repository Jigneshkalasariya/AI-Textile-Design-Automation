import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.response_models import UploadResponse
from app.services.storage_service import storage_service
from app.utils.file_utils import is_allowed_file, secure_filename
from app.core.logger import logger

router = APIRouter(prefix="/api")

@router.post(
    "/upload", 
    response_model=UploadResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Upload image file",
    description="Accepts a textile image file, sanitizes its filename, and stores it in the configured storage."
)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Filename is missing"
        )
        
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension. Allowed: PNG, JPG, JPEG, BMP, TIFF, SVG"
        )

    try:
        content = await file.read()
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        
        # Save to storage service
        saved_uri = storage_service.save_file(file_id, filename, content)
        
        logger.info(f"File uploaded successfully: {filename} (ID: {file_id}) saved to {saved_uri}")
        return UploadResponse(
            file_id=file_id,
            filename=filename,
            storage_type=storage_service.storage_type,
            message="Upload successful"
        )

    except Exception as e:
        logger.error(f"Error occurred during file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
