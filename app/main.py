from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logger import logger
from app.api.routes import upload, process, status as job_status, ai_analysis

# Initialize FastAPI App
app = FastAPI(
    title="AI Textile Design Automation Pipeline",
    description="Production-grade API for uploading, segmenting, and generating seamless textile repeat patterns for Texcelle.",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(job_status.router)
app.include_router(ai_analysis.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    logger.info(f"Configuration settings loaded successfully. Storage type: {settings.STORAGE_TYPE}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application...")

# Global exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception occurred on request {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the system administrator."}
    )

@app.get("/", summary="Health check root endpoint")
async def health_check():
    return {
        "status": "healthy",
        "app": "AI Textile Design Automation Platform",
        "storage": settings.STORAGE_TYPE,
        "debug_mode": settings.DEBUG
    }
