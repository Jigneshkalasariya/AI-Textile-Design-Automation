import cloudinary
import cloudinary.uploader
import cloudinary.api
from app.core.config import settings
from app.core.logger import logger

# Initialize Cloudinary configuration
def init_cloudinary():
    if not settings.CLOUDINARY_CLOUD_NAME or not settings.CLOUDINARY_API_KEY:
        logger.warning("Cloudinary credentials are not fully set. Cloudinary service might not work properly.")
        return

    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("Cloudinary configured successfully.")

def get_file_url(public_id: str) -> str:
    """
    Get the URL for a given Cloudinary public ID.
    """
    return cloudinary.CloudinaryImage(public_id).build_url()

def upload_image(file_path_or_url: str, folder: str = "ai_processed") -> dict:
    """
    Upload an image to Cloudinary and return the response dictionary.
    Supports large files > 10MB using upload_large.
    """
    try:
        response = cloudinary.uploader.upload_large(file_path_or_url, folder=folder, resource_type="auto")
        logger.info(f"Successfully uploaded image to Cloudinary: {response.get('secure_url')}")
        return response
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {e}")
        raise e
