import asyncio
import os
import wave
import uuid
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import aiofiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
def get_or_create_metric(metric_type, name, *args, **kwargs):
    """Get existing metric or create new one to avoid duplication during reload"""
    try:
        return metric_type(name, *args, **kwargs)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already exists, try to get it from the registry
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == name:
                    return collector
            # If we can't find it, just return None and let the app continue
            return None
        raise

# Initialize metrics with error handling
try:
    REQUEST_COUNT = Counter('music_generation_requests_total', 'Total music generation requests', ['status'])
    REQUEST_DURATION = Histogram('music_generation_duration_seconds', 'Music generation duration')
    API_KEY_USAGE = Counter('api_key_usage_total', 'API key usage count', ['key_index'])
except ValueError as e:
    if "Duplicated timeseries" in str(e):
        # Metrics already exist, use None placeholders
        REQUEST_COUNT = None
        REQUEST_DURATION = None
        API_KEY_USAGE = None
    else:
        raise

# Configuration
class Config:
    # API Keys
    API_KEYS: List[str] = []
    for i in range(1, 11):
        key = os.getenv(f'GENAI_API_KEY_{i}')
        if key:
            API_KEYS.append(key)
    
    if not API_KEYS:
        raise ValueError("No API keys found in environment variables")
    
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # CORS
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    # File Storage
    UPLOADS_DIR = Path(os.getenv('UPLOADS_DIR', 'uploads'))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 50))
    CLEANUP_OLD_FILES_DAYS = int(os.getenv('CLEANUP_OLD_FILES_DAYS', 7))
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', 60))
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_REQUESTS_PER_HOUR', 1000))
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    RETRY_DELAY_SECONDS = int(os.getenv('RETRY_DELAY_SECONDS', 5))
    API_KEY_ROTATION_ENABLED = os.getenv('API_KEY_ROTATION_ENABLED', 'true').lower() == 'true'
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')

config = Config()

# Create uploads directory
config.UPLOADS_DIR.mkdir(exist_ok=True)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# API Key Manager
class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.key_usage = {i: 0 for i in range(len(api_keys))}
        self.key_errors = {i: 0 for i in range(len(api_keys))}
        self.lock = asyncio.Lock()
    
    async def get_next_key(self) -> str:
        async with self.lock:
            key = self.api_keys[self.current_index]
            self.key_usage[self.current_index] += 1
            if API_KEY_USAGE is not None:
                API_KEY_USAGE.labels(key_index=self.current_index).inc()
            return key
    
    async def switch_key(self):
        async with self.lock:
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            logger.info(f"Switched to API key {self.current_index + 1}")
    
    async def mark_key_error(self, key_index: int):
        async with self.lock:
            self.key_errors[key_index] += 1
            logger.warning(f"API key {key_index + 1} error count: {self.key_errors[key_index]}")
    
    def get_key_stats(self):
        return {
            "total_keys": len(self.api_keys),
            "current_key": self.current_index + 1,
            "key_usage": self.key_usage,
            "key_errors": self.key_errors
        }

api_key_manager = APIKeyManager(config.API_KEYS)

# Gemini Client Manager
class GeminiClientManager:
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.clients = {}
    
    async def get_client(self) -> genai.Client:
        key = await self.api_key_manager.get_next_key()
        if key not in self.clients:
            self.clients[key] = genai.Client(
                api_key=key, 
                http_options={'api_version': 'v1alpha'}
            )
        return self.clients[key]
    
    async def switch_client(self):
        await self.api_key_manager.switch_key()
        # Clear clients cache to force new client creation
        self.clients.clear()

gemini_manager = GeminiClientManager(api_key_manager)

# Pydantic models
class MusicGenerationRequest(BaseModel):
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="Text description of the music you want to generate. Be descriptive about genre, mood, instruments, and style.",
        example="upbeat electronic dance music with synthesizers and drum beats"
    )
    duration_seconds: Optional[int] = Field(
        25, 
        ge=1, 
        le=300, 
        description="Duration of the generated music in seconds (1-300)",
        example=30
    )
    bpm: Optional[int] = Field(
        110, 
        ge=60, 
        le=200, 
        description="Beats per minute - tempo of the music (60-200)",
        example=120
    )
    temperature: Optional[float] = Field(
        1.2, 
        ge=0.1, 
        le=2.0, 
        description="Creativity level - higher values create more experimental music (0.1-2.0)",
        example=1.0
    )
    density: Optional[float] = Field(
        0.8, 
        ge=0.1, 
        le=1.0, 
        description="Musical complexity - higher values create more complex arrangements (0.1-1.0)",
        example=0.8
    )
    brightness: Optional[float] = Field(
        0.6, 
        ge=0.1, 
        le=1.0, 
        description="Overall brightness of the sound (0.1-1.0)",
        example=0.7
    )
    scale: Optional[str] = Field(
        "G_MAJOR_E_MINOR", 
        description="Musical scale for the composition",
        example="G_MAJOR_E_MINOR"
    )
    guidance: Optional[float] = Field(
        4.5, 
        ge=1.0, 
        le=10.0, 
        description="How closely the music follows the prompt (1.0-10.0)",
        example=4.5
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "upbeat electronic dance music with synthesizers and drum beats",
                "duration_seconds": 30,
                "bpm": 120,
                "temperature": 1.0,
                "density": 0.8,
                "brightness": 0.7,
                "scale": "G_MAJOR_E_MINOR",
                "guidance": 4.5
            }
        }

class MusicGenerationResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the music generation task")
    status: str = Field(..., description="Current status of the task")
    message: str = Field(..., description="Human-readable message about the task")
    estimated_duration: Optional[int] = Field(None, description="Estimated time to completion in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "8ebf9101-36aa-4148-aeb8-df85030e73c5",
                "status": "started",
                "message": "Music generation started. Use the task_id to check status and download the file.",
                "estimated_duration": 35
            }
        }

class MusicStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the music generation task")
    status: str = Field(..., description="Current status: 'generating', 'completed', or 'failed'")
    filename: Optional[str] = Field(None, description="Name of the generated audio file")
    download_url: Optional[str] = Field(None, description="URL to download the generated music")
    error: Optional[str] = Field(None, description="Error message if the task failed")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "8ebf9101-36aa-4148-aeb8-df85030e73c5",
                "status": "completed",
                "filename": "8ebf9101-36aa-4148-aeb8-df85030e73c5.wav",
                "download_url": "/download-music/8ebf9101-36aa-4148-aeb8-df85030e73c5",
                "progress_percentage": 100.0
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    message: str = Field(..., description="Health status message")
    api_keys_available: int = Field(..., description="Number of available API keys")
    uptime: str = Field(..., description="API uptime information")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Music generator API is running",
                "api_keys_available": 1,
                "uptime": "running",
                "version": "1.0.0"
            }
        }

class APIKeyStatsResponse(BaseModel):
    total_keys: int = Field(..., description="Total number of API keys configured")
    current_key: int = Field(..., description="Index of the currently active API key")
    key_usage: dict = Field(..., description="Usage count for each API key")
    key_errors: dict = Field(..., description="Error count for each API key")

    class Config:
        json_schema_extra = {
            "example": {
                "total_keys": 1,
                "current_key": 1,
                "key_usage": {"0": 5},
                "key_errors": {"0": 0}
            }
        }

# Task storage (in production, use Redis or database)
generation_tasks = {}

# Scale mapping
SCALE_MAP = {
    "G_MAJOR_E_MINOR": types.Scale.G_MAJOR_E_MINOR,
    "C_MAJOR_A_MINOR": types.Scale.C_MAJOR_A_MINOR,
    "F_MAJOR_D_MINOR": types.Scale.F_MAJOR_D_MINOR,
    "B_FLAT_MAJOR_G_MINOR": types.Scale.B_FLAT_MAJOR_G_MINOR,
    "E_FLAT_MAJOR_C_MINOR": types.Scale.E_FLAT_MAJOR_C_MINOR,
    "A_FLAT_MAJOR_F_MINOR": types.Scale.A_FLAT_MAJOR_F_MINOR,
    "D_FLAT_MAJOR_B_FLAT_MINOR": types.Scale.D_FLAT_MAJOR_B_FLAT_MINOR,
    "G_FLAT_MAJOR_E_FLAT_MINOR": types.Scale.G_FLAT_MAJOR_E_FLAT_MINOR,
    "B_MAJOR_A_FLAT_MINOR": types.Scale.B_MAJOR_A_FLAT_MINOR,
    "E_MAJOR_D_FLAT_MINOR": types.Scale.E_MAJOR_D_FLAT_MINOR,
    "A_MAJOR_G_FLAT_MINOR": types.Scale.A_MAJOR_G_FLAT_MINOR,
    "D_MAJOR_B_MINOR": types.Scale.D_MAJOR_B_MINOR,
}

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Sofy Music Generator API", 
                api_keys_count=len(config.API_KEYS),
                version="1.0.0")
    
    # Cleanup old files
    await cleanup_old_files()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sofy Music Generator API")

# Create FastAPI app
app = FastAPI(
    title="Sofy Music Generator API",
    description="""
    🎵 **Production-ready music generation API powered by Google's Lyria RealTime**

    ## Features
    - **AI Music Generation**: Create unique instrumental music from text descriptions
    - **Multiple API Keys**: Automatic rotation and retry logic for reliability
    - **Real-time Generation**: Stream music generation with WebSocket connections
    - **Production Ready**: Rate limiting, monitoring, and error handling

    ## How to Use
    1. **Generate Music**: POST `/generate-music` with your music description
    2. **Check Status**: GET `/music-status/{task_id}` to monitor progress
    3. **Download**: GET `/download-music/{task_id}` to get your audio file

    ## Supported Scales
    - G_MAJOR_E_MINOR (default)
    - C_MAJOR_A_MINOR
    - F_MAJOR_D_MINOR
    - B_FLAT_MAJOR_G_MINOR
    - E_FLAT_MAJOR_C_MINOR
    - A_FLAT_MAJOR_F_MINOR
    - D_FLAT_MAJOR_B_FLAT_MINOR
    - G_FLAT_MAJOR_E_FLAT_MINOR
    - B_MAJOR_A_FLAT_MINOR
    - E_MAJOR_D_FLAT_MINOR
    - A_MAJOR_G_FLAT_MINOR
    - D_MAJOR_B_MINOR

    ## Rate Limits
    - 60 requests per minute
    - 1000 requests per hour

    ## Audio Format
    - **Format**: WAV (16-bit PCM)
    - **Sample Rate**: 48kHz
    - **Channels**: Stereo (2 channels)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly in production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Utility functions
async def cleanup_old_files():
    """Clean up old music files"""
    cutoff_date = datetime.now() - timedelta(days=config.CLEANUP_OLD_FILES_DAYS)
    deleted_count = 0
    
    for file_path in config.UPLOADS_DIR.glob("*.wav"):
        if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete old file {file_path}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old files")

async def generate_music_with_retry(task_id: str, request: MusicGenerationRequest, retry_count: int = 0):
    """Generate music with retry logic and API key rotation"""
    start_time = time.time()
    
    try:
        client = await gemini_manager.get_client()
        current_key_index = api_key_manager.current_index
        
        logger.info(f"Starting music generation", 
                   task_id=task_id,
                   prompt=request.prompt,
                   api_key_index=current_key_index + 1,
                   retry_count=retry_count)
        
        filename = f"{task_id}.wav"
        filepath = config.UPLOADS_DIR / filename
        
        audio_chunks = []
        stop_event = asyncio.Event()

        async def receive_audio(session):
            try:
                async for message in session.receive():
                    if hasattr(message, 'server_content') and hasattr(message.server_content, 'audio_chunks'):
                        audio_data = message.server_content.audio_chunks[0].data
                        audio_chunks.append(audio_data)
                    if stop_event.is_set():
                        break
            except Exception as e:
                # Check if this is a normal WebSocket closure
                if "1000" in str(e) and "OK" in str(e):
                    logger.info(f"WebSocket closed normally after music generation", task_id=task_id)
                    return  # This is normal, not an error
                else:
                    logger.error(f"Audio receiver error: {e}")
                    raise

        selected_scale = SCALE_MAP.get(request.scale, types.Scale.G_MAJOR_E_MINOR)

        async with client.aio.live.music.connect(model='models/lyria-realtime-exp') as session:
            receiver_task = asyncio.create_task(receive_audio(session))

            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=request.prompt, weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    bpm=request.bpm,
                    temperature=request.temperature,
                    density=request.density,
                    brightness=request.brightness,
                    scale=selected_scale,
                    guidance=request.guidance,
                    top_k=100,
                    seed=42,
                    mute_bass=False,
                    mute_drums=False,
                    only_bass_and_drums=False,
                )
            )

            logger.info(f"Starting music generation", task_id=task_id)
            await session.play()
            await asyncio.sleep(request.duration_seconds)
            logger.info(f"Stopping music generation", task_id=task_id)
            await session.stop()
            await session.close()
            stop_event.set()
            await receiver_task

        # Combine and save audio
        pcm_data = b"".join(audio_chunks)
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_data)
        
        duration = time.time() - start_time
        if REQUEST_DURATION is not None:
            REQUEST_DURATION.observe(duration)
        
        logger.info(f"Music generation completed", 
                   task_id=task_id,
                   duration_seconds=duration,
                   file_size_bytes=len(pcm_data))
        
        generation_tasks[task_id] = {
            "status": "completed",
            "filename": filename,
            "error": None,
            "completed_at": datetime.now().isoformat()
        }
        
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(status="success").inc()
        
    except Exception as error:
        duration = time.time() - start_time
        if REQUEST_DURATION is not None:
            REQUEST_DURATION.observe(duration)
        
        error_message = str(error)
        logger.error(f"Music generation failed", 
                    task_id=task_id,
                    error=error_message,
                    retry_count=retry_count,
                    duration_seconds=duration)
        
        # Handle specific errors with retry logic
        if retry_count < config.MAX_RETRIES:
            # Check if this is a normal WebSocket closure (not an error)
            if "1000" in error_message and "OK" in error_message:
                logger.info(f"WebSocket closed normally, not retrying", task_id=task_id)
                # Mark as completed if we have audio data
                if audio_chunks:
                    pcm_data = b"".join(audio_chunks)
                    with wave.open(str(filepath), 'wb') as wf:
                        wf.setnchannels(2)
                        wf.setsampwidth(2)
                        wf.setframerate(48000)
                        wf.writeframes(pcm_data)
                    
                    generation_tasks[task_id] = {
                        "status": "completed",
                        "filename": filename,
                        "error": None,
                        "completed_at": datetime.now().isoformat()
                    }
                    
                    if REQUEST_COUNT is not None:
                        REQUEST_COUNT.labels(status="success").inc()
                    return
                else:
                    # No audio data, treat as error
                    pass
            
            elif any(keyword in error_message.lower() for keyword in ['429', 'quota', 'limit', 'rate']):
                logger.warning(f"Rate limit hit, switching API key", 
                             task_id=task_id,
                             current_key=api_key_manager.current_index + 1)
                await api_key_manager.mark_key_error(api_key_manager.current_index)
                await gemini_manager.switch_client()
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
                return await generate_music_with_retry(task_id, request, retry_count + 1)
            
            elif any(keyword in error_message.lower() for keyword in ['503', 'unavailable', 'timeout']):
                logger.warning(f"Service unavailable, retrying", 
                             task_id=task_id,
                             retry_count=retry_count + 1)
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
                return await generate_music_with_retry(task_id, request, retry_count + 1)
            
            else:
                logger.warning(f"Unknown error, retrying", 
                             task_id=task_id,
                             retry_count=retry_count + 1)
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
                return await generate_music_with_retry(task_id, request, retry_count + 1)
        
        # Max retries reached
        generation_tasks[task_id] = {
            "status": "failed",
            "filename": None,
            "error": error_message,
            "failed_at": datetime.now().isoformat()
        }
        
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(status="failed").inc()
        raise error

# API endpoints
@app.post("/generate-music", response_model=MusicGenerationResponse)
async def generate_music(
    request: MusicGenerationRequest, 
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    🎵 Generate music from a text description
    
    Creates a new music generation task using Google's Lyria RealTime AI model.
    The music generation runs in the background and you can monitor progress using the returned task_id.
    
    ## Process
    1. Validates the request parameters
    2. Creates a unique task ID
    3. Starts background music generation
    4. Returns task information for status tracking
    
    ## Example Request
    ```json
    {
        "prompt": "upbeat electronic dance music with synthesizers and drum beats",
        "duration_seconds": 30,
        "bpm": 120,
        "temperature": 1.0,
        "density": 0.8,
        "brightness": 0.7,
        "scale": "G_MAJOR_E_MINOR",
        "guidance": 4.5
    }
    ```
    
    ## Response
    - **task_id**: Use this to check status and download the file
    - **status**: Always "started" for new requests
    - **estimated_duration**: Approximate time to completion
    
    ## Next Steps
    1. Use the `task_id` to check status: `GET /music-status/{task_id}`
    2. Download the file when complete: `GET /download-music/{task_id}`
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task
        generation_tasks[task_id] = {
            "status": "generating",
            "filename": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
            "request": request.dict()
        }
        
        # Add to background tasks
        background_tasks.add_task(generate_music_with_retry, task_id, request)
        
        logger.info(f"Music generation task created", 
                   task_id=task_id,
                   prompt=request.prompt,
                   duration=request.duration_seconds)
        
        return MusicGenerationResponse(
            task_id=task_id,
            status="started",
            message="Music generation started. Use the task_id to check status and download the file.",
            estimated_duration=request.duration_seconds + 10  # Add buffer
        )
        
    except Exception as e:
        logger.error(f"Failed to start music generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start music generation: {str(e)}")

@app.get("/music-status/{task_id}", response_model=MusicStatusResponse)
async def get_music_status(task_id: str):
    """
    📊 Check the status of a music generation task
    
    Monitor the progress of your music generation request using the task_id returned from `/generate-music`.
    
    ## Status Types
    - **generating**: Music is currently being generated
    - **completed**: Music generation finished successfully
    - **failed**: Music generation failed with an error
    
    ## Response Fields
    - **task_id**: The unique task identifier
    - **status**: Current status of the task
    - **filename**: Name of the generated file (when completed)
    - **download_url**: URL to download the file (when completed)
    - **error**: Error message (if failed)
    - **progress_percentage**: Estimated progress (0-100)
    
    ## Example Response (Generating)
    ```json
    {
        "task_id": "8ebf9101-36aa-4148-aeb8-df85030e73c5",
        "status": "generating",
        "progress_percentage": 45.2
    }
    ```
    
    ## Example Response (Completed)
    ```json
    {
        "task_id": "8ebf9101-36aa-4148-aeb8-df85030e73c5",
        "status": "completed",
        "filename": "8ebf9101-36aa-4148-aeb8-df85030e73c5.wav",
        "download_url": "/download-music/8ebf9101-36aa-4148-aeb8-df85030e73c5",
        "progress_percentage": 100.0
    }
    ```
    """
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = generation_tasks[task_id]
    
    if task_info["status"] == "completed":
        download_url = f"/download-music/{task_id}"
        return MusicStatusResponse(
            task_id=task_id,
            status=task_info["status"],
            filename=task_info["filename"],
            download_url=download_url,
            progress_percentage=100.0
        )
    elif task_info["status"] == "failed":
        return MusicStatusResponse(
            task_id=task_id,
            status=task_info["status"],
            error=task_info["error"]
        )
    else:
        # Calculate progress based on time elapsed
        started_at = datetime.fromisoformat(task_info["started_at"])
        elapsed = (datetime.now() - started_at).total_seconds()
        estimated_duration = task_info["request"]["duration_seconds"] + 10
        progress = min(95.0, (elapsed / estimated_duration) * 100)
        
        return MusicStatusResponse(
            task_id=task_id,
            status=task_info["status"],
            progress_percentage=progress
        )

@app.get("/download-music/{task_id}")
async def download_music(task_id: str):
    """
    📥 Download the generated music file
    
    Download your completed music generation as a WAV audio file.
    
    ## Requirements
    - Task must be in "completed" status
    - Use the task_id from `/generate-music` or `/music-status/{task_id}`
    
    ## File Format
    - **Format**: WAV (16-bit PCM)
    - **Sample Rate**: 48kHz
    - **Channels**: Stereo (2 channels)
    - **Filename**: `{task_id}.wav`
    
    ## Response
    - **Content-Type**: `audio/wav`
    - **Content-Disposition**: `attachment; filename="{task_id}.wav"`
    - **Body**: Binary audio data
    
    ## Error Codes
    - **404**: Task not found or file not found
    - **400**: Music generation not completed yet
    
    ## Usage
    ```bash
    curl -O -J "http://localhost:8000/download-music/8ebf9101-36aa-4148-aeb8-df85030e73c5"
    ```
    """
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = generation_tasks[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Music generation not completed")
    
    filename = task_info["filename"]
    filepath = config.UPLOADS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Music file not found")
    
    logger.info(f"Music file downloaded", task_id=task_id, filename=filename)
    
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="audio/wav"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    🏥 Health check endpoint
    
    Check the overall health and status of the music generation API.
    
    ## Response Fields
    - **status**: "healthy" if the API is running properly
    - **message**: Human-readable status message
    - **api_keys_available**: Number of configured API keys
    - **uptime**: API uptime information
    - **version**: Current API version
    
    ## Example Response
    ```json
    {
        "status": "healthy",
        "message": "Music generator API is running",
        "api_keys_available": 1,
        "uptime": "running",  # In production, track actual uptime
        "version": "1.0.0"
    }
    ```
    
    ## Use Cases
    - **Load Balancer Health Checks**: Monitor API availability
    - **System Monitoring**: Track API status and configuration
    - **Development**: Verify API is running during development
    """
    return HealthResponse(
        status="healthy",
        message="Music generator API is running",
        api_keys_available=len(config.API_KEYS),
        uptime="running",  # In production, track actual uptime
        version="1.0.0"
    )

@app.get("/metrics")
async def metrics():
    """
    📊 Prometheus metrics endpoint
    
    Expose Prometheus-formatted metrics for monitoring and observability.
    
    ## Metrics Available
    - **music_generation_requests_total**: Total number of music generation requests
    - **music_generation_duration_seconds**: Duration of music generation tasks
    - **api_key_usage_total**: Usage count for each API key
    - **python_gc_***: Python garbage collection metrics
    
    ## Response Format
    - **Content-Type**: `text/plain; version=0.0.4; charset=utf-8`
    - **Body**: Prometheus exposition format
    
    ## Example Usage
    ```bash
    curl http://localhost:8000/metrics
    ```
    
    ## Integration
    - **Prometheus**: Scrape this endpoint for metrics collection
    - **Grafana**: Visualize metrics in dashboards
    - **Alerting**: Set up alerts based on metric thresholds
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api-keys/stats", response_model=APIKeyStatsResponse)
async def api_key_stats():
    """
    🔑 API key usage statistics
    
    Get detailed statistics about API key usage, rotation, and error rates.
    
    ## Response Fields
    - **total_keys**: Total number of configured API keys
    - **current_key**: Index of the currently active API key (1-based)
    - **key_usage**: Dictionary mapping key index to usage count
    - **key_errors**: Dictionary mapping key index to error count
    
    ## Example Response
    ```json
    {
        "total_keys": 1,
        "current_key": 1,
        "key_usage": {"0": 5},
        "key_errors": {"0": 0}
    }
    ```
    
    ## Use Cases
    - **Monitoring**: Track API key usage patterns
    - **Debugging**: Identify problematic API keys
    - **Load Balancing**: Understand key rotation behavior
    - **Billing**: Monitor API usage for cost management
    """
    return api_key_manager.get_key_stats()

@app.get("/")
async def root():
    """
    🏠 API root endpoint
    
    Welcome page with API information and available endpoints.
    
    ## Response Fields
    - **message**: API name and description
    - **version**: Current API version
    - **status**: API deployment status
    - **endpoints**: Dictionary of available endpoints with their HTTP methods
    
    ## Example Response
    ```json
    {
        "message": "Sofy Music Generator API",
        "version": "1.0.0",
        "status": "production",
        "endpoints": {
            "generate_music": "POST /generate-music",
            "check_status": "GET /music-status/{task_id}",
            "download_music": "GET /download-music/{task_id}",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "api_key_stats": "GET /api-keys/stats"
        }
    }
    ```
    
    ## Quick Start
    1. Check API health: `GET /health`
    2. Generate music: `POST /generate-music`
    3. Monitor progress: `GET /music-status/{task_id}`
    4. Download result: `GET /download-music/{task_id}`
    
    ## Documentation
    - **Swagger UI**: `/docs` - Interactive API documentation
    - **ReDoc**: `/redoc` - Alternative documentation view
    - **OpenAPI JSON**: `/openapi.json` - Raw OpenAPI specification
    """
    return {
        "message": "Sofy Music Generator API",
        "version": "1.0.0",
        "status": "production",
        "endpoints": {
            "generate_music": "POST /generate-music",
            "check_status": "GET /music-status/{task_id}",
            "download_music": "GET /download-music/{task_id}",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "api_key_stats": "GET /api-keys/stats"
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception", 
                path=request.url.path,
                method=request.method,
                error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    ) 