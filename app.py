import asyncio
import os
import wave
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Annotated
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, BeforeValidator
from bson import ObjectId
import motor.motor_asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from dotenv import load_dotenv

from google import genai
from google.genai import types

# This is a Pydantic V2 helper to integrate MongoDB's ObjectId
PyObjectId = Annotated[ObjectId, BeforeValidator(lambda v: ObjectId(v) if ObjectId.is_valid(v) else v)]

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
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, *args, **kwargs)

REQUEST_COUNT = get_or_create_metric(Counter, 'music_generation_requests_total', 'Total music generation requests', ['status'])
REQUEST_DURATION = get_or_create_metric(Histogram, 'music_generation_duration_seconds', 'Music generation duration')
API_KEY_USAGE = get_or_create_metric(Counter, 'api_key_usage_total', 'API key usage count', ['key_index'])

# Configuration
class Config:
    def __init__(self):
        # API Keys
        self.API_KEYS = []
        for i in range(1, 11):
            key = os.getenv(f'GENAI_API_KEY_{i}')
            if key:
                self.API_KEYS.append(key)
        
        if not self.API_KEYS:
            raise ValueError("No API keys found. Set GENAI_API_KEY_1, etc.")
        
        # MongoDB
        self.MONGO_URI = os.getenv('MONGO_URI', 'mongodb://mongodb:27017')
        self.MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'music_generation')
        
        # Server
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', 8000))
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        self.ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
        
        self.UPLOADS_DIR = Path(os.getenv('UPLOADS_DIR', 'uploads'))
        
        self.RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', 60))
        
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
        self.RETRY_DELAY_SECONDS = int(os.getenv('RETRY_DELAY_SECONDS', 5))

config = Config()
config.UPLOADS_DIR.mkdir(exist_ok=True)

# Database
class Database:
    def __init__(self, uri, db_name):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.tasks = self.db.get_collection("tasks")
        self.api_key_stats = self.db.get_collection("api_key_stats")

    def close(self):
        self.client.close()

db_client: Optional[Database] = None

def get_db():
    global db_client
    if db_client is None:
        db_client = Database(config.MONGO_URI, config.MONGO_DB_NAME)
    return db_client

# API Key Manager
class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def get_next_key(self, db: Database) -> str:
        async with self.lock:
            key_index = self.current_index
            key = self.api_keys[key_index]
            await db.api_key_stats.update_one(
                {'key_index': key_index},
                {'$inc': {'usage_count': 1}, '$setOnInsert': {'key_index': key_index}},
                upsert=True
            )
            API_KEY_USAGE.labels(key_index=key_index).inc()
            return key
    
    async def switch_key(self):
        async with self.lock:
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            logger.info(f"Switched to API key index {self.current_index}")
    
    async def mark_key_error(self, db: Database, key_index: int):
        await db.api_key_stats.update_one(
            {'key_index': key_index},
            {'$inc': {'error_count': 1}, '$setOnInsert': {'key_index': key_index}},
            upsert=True
        )
        logger.warning(f"API key index {key_index} error count incremented.")
    
    async def get_key_stats(self, db: Database) -> Dict:
        stats_cursor = db.api_key_stats.find()
        stats = await stats_cursor.to_list(length=len(self.api_keys))
        
        key_usage = {str(i): 0 for i in range(len(self.api_keys))}
        key_errors = {str(i): 0 for i in range(len(self.api_keys))}
        
        for stat in stats:
            key_usage[str(stat['key_index'])] = stat.get('usage_count', 0)
            key_errors[str(stat['key_index'])] = stat.get('error_count', 0)
            
        return {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_index,
            "key_usage": key_usage,
            "key_errors": key_errors
        }

api_key_manager = APIKeyManager(config.API_KEYS)

# Gemini Client Manager
class GeminiClientManager:
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.clients = {}
    
    async def get_client(self, db: Database) -> genai.Client:
        key = await self.api_key_manager.get_next_key(db)
        if key not in self.clients:
            self.clients[key] = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})
        return self.clients[key]
    
    async def switch_client(self):
        await self.api_key_manager.switch_key()
        self.clients.clear()

gemini_manager = GeminiClientManager(api_key_manager)

# Pydantic Models
class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, example="upbeat electronic dance music")
    duration_seconds: int = Field(25, ge=1, le=300)
    bpm: int = Field(110, ge=60, le=200)
    temperature: float = Field(1.2, ge=0.1, le=2.0)
    density: float = Field(0.8, ge=0.1, le=1.0)
    brightness: float = Field(0.6, ge=0.1, le=1.0)
    scale: str = Field("G_MAJOR_E_MINOR", description="Musical scale")
    guidance: float = Field(4.5, ge=1.0, le=10.0)

class TaskInDB(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    status: str = "generating"
    filename: Optional[str] = None
    error: Optional[str] = None
    request_data: dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MusicGenerationResponse(BaseModel):
    task_id: str
    status: str = "started"
    message: str = "Music generation started."

class MusicStatusResponse(BaseModel):
    task_id: str
    status: str
    filename: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None

class APIKeyStatsResponse(BaseModel):
    total_keys: int
    current_key_index: int
    key_usage: Dict[str, int]
    key_errors: Dict[str, int]

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = get_db()
    logger.info("MongoDB connection setup.")
    yield
    app.state.db.close()
    logger.info("MongoDB connection closed.")

app = FastAPI(title="Sofy Music Generator API", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.state.limiter = Limiter(key_func=get_remote_address)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

async def generate_music_task(task_id_str: str, request: MusicGenerationRequest, db: Database):
    task_id = ObjectId(task_id_str)
    start_time = time.time()
    
    for attempt in range(config.MAX_RETRIES):
        try:
            client = await gemini_manager.get_client(db)
            current_key_index = api_key_manager.current_index
            logger.info("Starting music generation", task_id=task_id_str, attempt=attempt + 1, key_index=current_key_index)

            filename = f"{task_id_str}.wav"
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
                    if "1000" not in str(e):
                        logger.error(f"Audio receiver error: {e}", task_id=task_id_str)
                        raise

            selected_scale = getattr(types.Scale, request.scale.upper(), types.Scale.G_MAJOR_E_MINOR)

            async with client.aio.live.music.connect(model='models/lyria-realtime-exp') as session:
                receiver_task = asyncio.create_task(receive_audio(session))
                await session.set_weighted_prompts(prompts=[types.WeightedPrompt(text=request.prompt, weight=1.0)])
                await session.set_music_generation_config(
                    config=types.LiveMusicGenerationConfig(
                        bpm=request.bpm, temperature=request.temperature, density=request.density,
                        brightness=request.brightness, scale=selected_scale, guidance=request.guidance,
                    )
                )
                await session.play()
                await asyncio.sleep(request.duration_seconds)
                await session.stop()
                await session.close()
                stop_event.set()
                await receiver_task

            pcm_data = b"".join(audio_chunks)
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_data)
            
            await db.tasks.update_one(
                {"_id": task_id},
                {"$set": {"status": "completed", "filename": filename, "completed_at": datetime.utcnow()}}
            )
            REQUEST_COUNT.labels(status="success").inc()
            REQUEST_DURATION.observe(time.time() - start_time)
            logger.info("Music generation completed", task_id=task_id_str, duration=time.time() - start_time)
            return

        except Exception as e:
            logger.error("Music generation failed", task_id=task_id_str, error=str(e), attempt=attempt + 1)
            if attempt < config.MAX_RETRIES - 1:
                await api_key_manager.mark_key_error(db, api_key_manager.current_index)
                await gemini_manager.switch_client()
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
            else:
                await db.tasks.update_one(
                    {"_id": task_id},
                    {"$set": {"status": "failed", "error": str(e), "completed_at": datetime.utcnow()}}
                )
                REQUEST_COUNT.labels(status="failed").inc()
                REQUEST_DURATION.observe(time.time() - start_time)
                return

@app.post("/generate-music", response_model=MusicGenerationResponse, status_code=202)
async def generate_music(request: MusicGenerationRequest, background_tasks: BackgroundTasks, request_obj: Request):
    db = request_obj.app.state.db
    task = TaskInDB(request_data=request.model_dump())
    result = await db.tasks.insert_one(task.model_dump(by_alias=True))
    task_id = str(result.inserted_id)
    background_tasks.add_task(generate_music_task, task_id, request, db)
    return MusicGenerationResponse(task_id=task_id)

@app.get("/music-status/{task_id}", response_model=MusicStatusResponse)
async def get_music_status(task_id: str, request: Request):
    db = request.app.state.db
    if not ObjectId.is_valid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    task = await db.tasks.find_one({"_id": ObjectId(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = TaskInDB.model_validate(task)
    response = MusicStatusResponse(task_id=str(task_data.id), status=task_data.status)
    if task_data.status == 'completed':
        response.filename = task_data.filename
        response.download_url = f"/download-music/{task_id}"
    elif task_data.status == 'failed':
        response.error = task_data.error
    return response

@app.get("/download-music/{task_id}")
async def download_music(task_id: str, request: Request):
    db = request.app.state.db
    if not ObjectId.is_valid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    task = await db.tasks.find_one({"_id": ObjectId(task_id)})
    
    if not task or task.get('status') != 'completed' or not task.get('filename'):
        raise HTTPException(status_code=404, detail="Task not complete or file not found")

    filepath = config.UPLOADS_DIR / task['filename']
    if not filepath.exists():
        logger.error("File not found on server though task is complete", task_id=task_id, filepath=str(filepath))
        raise HTTPException(status_code=404, detail="File not found on server")
    
    return FileResponse(path=str(filepath), filename=task['filename'], media_type="audio/wav")

@app.get("/api-keys/stats", response_model=APIKeyStatsResponse)
async def get_api_key_stats(request: Request):
    db = request.app.state.db
    stats = await api_key_manager.get_key_stats(db)
    return APIKeyStatsResponse(**stats)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.1.0"}
    
@app.get("/")
async def root():
    return {"message": "Sofy Music Generator API", "version": "1.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)