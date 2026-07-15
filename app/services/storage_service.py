import os
from pathlib import Path
from typing import List, Union
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
from app.core.logger import logger

class StorageService:
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.local_dir = settings.local_storage_path
        self.s3_client = None

        if self.storage_type == "s3":
            try:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    endpoint_url=settings.AWS_ENDPOINT_URL
                )
                logger.info(f"S3 Storage service initialized. Bucket: {settings.S3_BUCKET}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}. Falling back to LOCAL storage.")
                self.storage_type = "local"

    def save_file(self, file_id: str, filename: str, content: bytes) -> str:
        """
        Saves a file in the configured storage.
        Returns the reference/URI path.
        """
        if self.storage_type == "s3" and self.s3_client:
            s3_key = f"{file_id}/{filename}"
            try:
                self.s3_client.put_object(
                    Bucket=settings.S3_BUCKET,
                    Key=s3_key,
                    Body=content
                )
                logger.info(f"Saved file to S3: {s3_key}")
                return f"s3://{settings.S3_BUCKET}/{s3_key}"
            except ClientError as e:
                logger.error(f"S3 put_object failed: {e}. Falling back to save locally.")
                # Fallback to local
        
        # Local storage fallback
        dest_dir = self.local_dir / file_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename
        with open(dest_file, "wb") as f:
            f.write(content)
        logger.info(f"Saved file locally: {dest_file}")
        return str(dest_file)

    def get_file(self, file_id: str, filename: str) -> bytes:
        """Retrieves raw file bytes."""
        if self.storage_type == "s3" and self.s3_client:
            s3_key = f"{file_id}/{filename}"
            try:
                response = self.s3_client.get_object(
                    Bucket=settings.S3_BUCKET,
                    Key=s3_key
                )
                return response["Body"].read()
            except ClientError as e:
                logger.error(f"S3 get_object failed: {e}. Checking local fallback.")

        # Local read
        filepath = self.local_dir / file_id / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} not found for job/file_id {file_id}")
        with open(filepath, "rb") as f:
            return f.read()

    def get_local_path(self, file_id: str, filename: str) -> Path:
        """
        Ensure file is downloaded/available locally for processing and return local Path.
        """
        local_path = self.local_dir / file_id / filename
        if local_path.exists():
            return local_path

        # If it's on S3 and doesn't exist locally, download it
        if self.storage_type == "s3" and self.s3_client:
            s3_key = f"{file_id}/{filename}"
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(
                    settings.S3_BUCKET,
                    s3_key,
                    str(local_path)
                )
                logger.info(f"Downloaded S3 key {s3_key} to local path {local_path}")
                return local_path
            except ClientError as e:
                logger.error(f"Failed downloading S3 file: {e}")
                
        raise FileNotFoundError(f"File {filename} not found locally or on S3 for file_id {file_id}")

    def list_files(self, file_id: str) -> List[str]:
        """List filenames generated/stored under a specific file/job id."""
        filenames = []
        # Check local folder first
        local_job_dir = self.local_dir / file_id
        if local_job_dir.exists():
            filenames.extend([f.name for f in local_job_dir.iterdir() if f.is_file()])

        # If S3, search there as well
        if self.storage_type == "s3" and self.s3_client:
            prefix = f"{file_id}/"
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=settings.S3_BUCKET,
                    Prefix=prefix
                )
                if "Contents" in response:
                    for obj in response["Contents"]:
                        key = obj["Key"]
                        name = key.replace(prefix, "", 1)
                        if name and name not in filenames:
                            filenames.append(name)
            except ClientError as e:
                logger.error(f"S3 list_objects failed: {e}")
                
        return filenames

storage_service = StorageService()
