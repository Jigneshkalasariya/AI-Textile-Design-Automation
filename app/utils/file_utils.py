import os
import re
from pathlib import Path
from typing import Set

ALLOWED_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".psd", ".svg"
}

def is_allowed_file(filename: str) -> bool:
    """Check if the file has a supported image extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def secure_filename(filename: str) -> str:
    """
    Sanitize the filename by removing path characters and non-alphanumeric/dot characters.
    Similar to Werkzeug's secure_filename.
    """
    name = Path(filename).name
    # Replace whitespace with underscore
    name = re.sub(r"\s+", "_", name)
    # Remove characters that are not alphanumeric, dot, dash, or underscore
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    # Avoid empty names or directory traversal attempts
    if not name or name in {".", ".."}:
        return "uploaded_file.png"
    return name

def get_file_size_mb(filepath: Path) -> float:
    """Get file size in Megabytes."""
    if not filepath.exists():
        return 0.0
    return os.path.getsize(filepath) / (1024 * 1024)
