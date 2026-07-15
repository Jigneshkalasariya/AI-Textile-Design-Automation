# AI Textile Design Automation Platform - Backend Pipeline

This is a production-grade Python repository for an AI-powered Textile Design Automation Pipeline. The system is designed to ingest textile motif/design images, apply AI-based enhancements and segmentation steps, and output print-ready and engraving-ready formats (TIFF, BMP, PSD, PNG, SVG) suitable for CAD software such as Texcelle.

---

## System Purpose & Design

The platform processes textile design images through an asynchronous 11-step pipeline. To ensure ease of setup and robust execution in resource-constrained environments, all GPU-heavy ML operations (rembg background removal, YOLOv8 motif detection, and Stable Diffusion inpainting) have high-quality, zero-dependency OpenCV, NumPy, and Pillow fallbacks.

```
                  ┌──────────────────────────────┐
                  │      FastAPI Web Server      │
                  └──────────────┬───────────────┘
                                 │ Queue Job
                                 ▼
                  ┌──────────────────────────────┐
                  │         Redis Broker         │
                  └──────────────┬───────────────┘
                                 │ Dispatch Task
                                 ▼
                  ┌──────────────────────────────┐
                  │        Celery Worker         │
                  │  (Executes 11-Step Pipeline) │
                  └──────────────┬───────────────┘
                                 │ Read / Write
                                 ▼
                  ┌──────────────────────────────┐
                  │ Storage: Local / Cloud S3/R2 │
                  └──────────────────────────────┘
```

---

## 11-Step Processing Pipeline

1. **Image Enhancement**: Edge-preserving bilateral filtering (denoising) and unsharp masking (sharpening) with RealESRGAN super-resolution hooks.
2. **Background Removal**: Isolates motifs using `rembg` (U2Net) or an adaptive color-distance GrabCut fallback.
3. **Object Detection**: Identifies discrete motifs using YOLOv8 or high-speed contour bounding boxes.
4. **Pattern Detection**: Uses structural autocorrelation (2D template matching) to detect repeat boundaries (width/height) and repeat types (straight vs. half-drop).
5. **Pattern Repair (Inpainting)**: Applies Stable Diffusion boundary inpainting or fast marching `cv2.inpaint` algorithms to create seamless transitions along repeat borders.
6. **Color Separation**: Segments pixels into $K$ discrete layers using K-Means clustering in the perceptually uniform LAB color space.
7. **Color Reduction**: Quantizes the design to a production-size palette using Floyd-Steinberg dithering.
8. **Vectorization**: Traces raster boundaries using Douglas-Peucker polygon approximation to output optimized SVG paths.
9. **Repeat Generation**: Creates a tiled grid of repeat motifs matching straight or half-drop alignments.
10. **Color Variants (Colorways)**: Generates alternative colorways (e.g., complementary, analogous, triad) in the HSV color space.
11. **Output Generation**: Writes Texcelle-ready files: LZW-compressed TIFFs, flat BMPs, PSD files, PNGs, and SVGs.

---

## Project Structure

```
backend-python/
├── app/
│   ├── main.py                 # FastAPI application root
│   ├── core/
│   │   ├── config.py           # Configuration parsing (Pydantic settings)
│   │   └── logger.py           # Loguru custom logging configuration
│   ├── api/
│   │   └── routes/
│   │       ├── upload.py       # POST /api/upload (Saves original image)
│   │       ├── process.py      # POST /api/process/{file_id} (Triggers Celery task)
│   │       └── status.py       # GET /api/status/{job_id} & /api/download/...
│   ├── services/
│   │   ├── storage_service.py  # Local/S3 storage unified layer
│   │   ├── queue_service.py    # Celery task status monitoring
│   │   └── pipeline_service.py # Core orchestrator executing the 11 steps
│   ├── pipeline/
│   │   └── steps/              # The 11 modular steps
│   ├── workers/
│   │   ├── celery_worker.py    # Celery application entrypoint
│   │   └── tasks.py            # Celery task definitions (process_textile_image)
│   ├── models/
│   │   ├── request_models.py   # Step configuration schemas
│   │   └── response_models.py  # Job metadata response schemas
│   └── utils/
│       ├── image_utils.py      # Format conversions and synthetic image gen
│       └── file_utils.py       # Filename sanitization and checks
├── requirements.txt            # Pinned stable package requirements
├── pyproject.toml              # Tool configurations
├── .env.example                # Template environment variables
├── Dockerfile                  # Container instructions (Multi-stage slim image)
├── docker-compose.yml          # Local orchestration (API, worker, Redis)
├── test_pipeline.py            # Direct pipeline validation script
└── README.md                   # Setup guide
```

---

## Quick Start (No Infrastructure Required)

You can verify and execute the entire 11-step pipeline locally on a dynamically generated synthetic image in **offline fallback mode** using only a local python environment (no Redis, Docker, or AWS required):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the standalone pipeline validator
python test_pipeline.py
```
This script will:
- Generate a synthetic motif with colors.
- Run all 11 steps in sequence on your CPU.
- Save output TIFF, BMP, PSD, PNG, and SVG files under `./data/storage/test_pipeline_run/`.

---

## Docker Compose Deployment

To spin up the production-grade services locally (FastAPI app, Redis Broker, and Celery Worker):

```bash
# 1. Build and start services
docker-compose up --build

# 2. Check service status
docker-compose ps
```

The FastAPI Swagger Documentation will be accessible at: `http://localhost:8000/docs`

---

## API Documentation & Usage Flow

### 1. Upload Source Image
Uploads an image.
* **Endpoint**: `POST /api/upload`
* **Request**: Multipart Form Data (`file: file_stream`)
* **Response**:
```json
{
  "file_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "filename": "vintage_floral.png",
  "storage_type": "local",
  "message": "Upload successful"
}
```

```bash
curl -X POST -F "file=@your_design.png" http://localhost:8000/api/upload
```

### 2. Trigger Processing
Starts the pipeline task for the uploaded image.
* **Endpoint**: `POST /api/process/{file_id}`
* **Request Body**: (Optional JSON to custom-configure individual pipeline steps)
```json
{
  "color_separation": {
    "num_colors": 6,
    "color_space": "LAB"
  },
  "repeat_generation": {
    "repeat_type": "half-drop",
    "horizontal_tiles": 3,
    "vertical_tiles": 3
  }
}
```
* **Response**:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "PENDING",
  "message": "Processing job triggered successfully"
}
```

```bash
curl -X POST -H "Content-Type: application/json" -d '{"color_separation":{"num_colors":6}}' http://localhost:8000/api/process/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```

### 3. Track Status
Monitors progress.
* **Endpoint**: `GET /api/status/{job_id}`
* **Response**:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "PROCESSING",
  "progress": 70.0,
  "current_step": "color_reduction",
  "message": "Reducing colors to production palette...",
  "output_files": [],
  "error": null
}
```

When completed:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "SUCCESS",
  "progress": 100.0,
  "current_step": "complete",
  "message": "Pipeline completed successfully.",
  "output_files": [
    "vintage_floral_repeat.png",
    "vintage_floral_repeat.tif",
    "vintage_floral_repeat.bmp",
    "vintage_floral_repeat.svg",
    "vintage_floral_repeat.psd",
    "vintage_floral_colorway_complementary.png",
    "vintage_floral_colorway_analogous.png",
    "vintage_floral_layer_0_efefef.png",
    "vintage_floral_layer_1_5f8a20.png"
  ],
  "error": null
}
```

```bash
curl http://localhost:8000/api/status/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```

### 4. Download Output Files
Retrieve individual assets.
* **Endpoint**: `GET /api/download/{job_id}/{filename}`

```bash
curl -O http://localhost:8000/api/download/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d/vintage_floral_repeat.tif
```
