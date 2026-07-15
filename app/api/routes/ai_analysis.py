import base64
import mimetypes
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from app.services.storage_service import storage_service
from app.services.openrouter_service import openrouter_service
from app.utils.file_utils import is_allowed_file
from app.core.logger import logger

router = APIRouter(prefix="/api/ai")

@router.post(
    "/analyze/{file_id}",
    status_code=status.HTTP_200_OK,
    summary="Analyze uploaded design using OpenRouter AI",
    description="Uses OpenRouter Vision-LLM models to analyze an uploaded textile design image for motifs, defects, and repeat boundaries."
)
async def analyze_design(file_id: str, model: Optional[str] = None) -> Dict[str, Any]:
    # 1. Locate original uploaded file
    files = storage_service.list_files(file_id)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded files found for ID {file_id}"
        )
    
    original_file = None
    for f in files:
        if is_allowed_file(f) and not ("_repeat" in f or "_layer_" in f or "colorway_" in f or "version" in f or "sketch" in f):
            original_file = f
            break
            
    if not original_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original source image file not found in storage."
        )

    # 2. Get local file path and read bytes
    try:
        local_path = storage_service.get_local_path(file_id, original_file)
        with open(local_path, "rb") as image_file:
            image_data = image_file.read()
            
        # 3. Base64 encode the image
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        
        # 4. Resolve MIME type
        mime_type, _ = mimetypes.guess_type(original_file)
        if not mime_type:
            mime_type = "image/png"
            
        logger.info(f"Triggering OpenRouter Vision analysis for {original_file} (MIME: {mime_type})")
        
        # 5. Call OpenRouter service
        analysis_result = openrouter_service.analyze_design(
            image_base64=base64_encoded,
            mime_type=mime_type,
            model=model
        )
        
        return {
            "status": "SUCCESS",
            "file_id": file_id,
            "filename": original_file,
            "analysis": analysis_result
        }
        
    except ValueError as val_err:
        logger.error(f"Configuration error for OpenRouter: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as e:
        logger.exception(f"Failed during AI analysis of design: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze textile design image: {str(e)}"
        )
