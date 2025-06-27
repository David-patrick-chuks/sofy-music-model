# Sofy Music Generator API

> **IMPORTANT FOR RENDER DEPLOYMENT:**
> You MUST manually add all environment variables from your `.env` or `.env.example` file to the Render dashboard under the Environment tab. Render does NOT automatically use your local `.env` file. If you skip this step, your deployment will fail.

<p align="center">
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge" alt="Code style: black"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT"></a>
</p>

> [!DANGER]
> **IMPORTANT: Gemini Premium API Key Required**
>
> This project uses the Lyria RealTime model, which requires a **Gemini Premium API key** with billing enabled. Standard or free-tier keys will **not** work and will result in authentication or quota errors. Please ensure your API keys have the necessary permissions before deploying.

A production-ready FastAPI server that generates instrumental music from text prompts using Google's Gemini API with Lyria RealTime technology. Features multiple API key rotation, retry logic, and comprehensive monitoring.

## 🚀 Quick Deploy on Render

### 1. **Fork this Repository**
Fork this repository to your own GitHub account.

### 2. **Deploy with Docker on Render**
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sofy-ai/music-generation-server-template)

1. Go to the [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Connect the GitHub repository you forked.
4. Configure the service:
   - **Name**: `sofy-music-api` (or your preferred name)
   - **Environment**: Select `Docker`
   - **Health Check Path**: `/health`
5. Render will automatically detect and use your `Dockerfile`. No build or start command is needed.

### 3. **Set Environment Variables**
In the Render dashboard, go to the **Environment** tab and add the following variables.

**Required API Keys (at least one):**
```
GENAI_API_KEY_1=your_first_gemini_premium_key
GENAI_API_KEY_2=your_second_gemini_premium_key
# ... add up to 10 keys
```

**Optional Server Configuration:**
```
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
MAX_RETRIES=3
RETRY_DELAY_SECONDS=5
```

### 4. **Deploy**
Click **Create Web Service**. Render will build the Docker image and deploy your application. Your API will be live at the URL provided by Render.

## 📖 API Documentation

### Base URL
```
https://your-app-name.onrender.com
```

### Interactive API Docs
Visit `https://your-app-name.onrender.com/docs` for Swagger UI documentation.

### Endpoints

#### 1. Generate Music
```http
POST /generate-music
```

**Request Body:**
```json
{
  "prompt": "Afrobeat with smooth pianos and drums",
  "duration_seconds": 25,
  "bpm": 110,
  "temperature": 1.2,
  "density": 0.8,
  "brightness": 0.6,
  "scale": "G_MAJOR_E_MINOR",
  "guidance": 4.5
}
```

**Response:**
```json
{
  "task_id": "8e4a2e66-dd6a-4770-be64-e165a8485e4d",
  "status": "started",
  "message": "Music generation started. Use the task_id to check status and download the file.",
  "estimated_duration": 35
}
```

#### 2. Check Status
```http
GET /music-status/{task_id}
```

**Response (generating):**
```json
{
  "task_id": "8e4a2e66-dd6a-4770-be64-e165a8485e4d",
  "status": "generating",
  "progress_percentage": 45.2
}
```

**Response (completed):**
```json
{
  "task_id": "8e4a2e66-dd6a-4770-be64-e165a8485e4d",
  "status": "completed",
  "filename": "8e4a2e66-dd6a-4770-be64-e165a8485e4d.wav",
  "download_url": "/download-music/8e4a2e66-dd6a-4770-be64-e165a8485e4d",
  "progress_percentage": 100.0
}
```

#### 3. Download Music
```http
GET /download-music/{task_id}
```

Returns the generated WAV file for download.

#### 4. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Music generator API is running",
  "api_keys_available": 10,
  "uptime": "running",
  "version": "1.0.0"
}
```

#### 5. API Key Statistics
```http
GET /api-keys/stats
```

**Response:**
```json
{
  "total_keys": 10,
  "current_key": 3,
  "key_usage": {"0": 15, "1": 12, "2": 8, ...},
  "key_errors": {"0": 0, "1": 1, "2": 0, ...}
}
```

#### 6. Metrics (Prometheus)
```http
GET /metrics
```

Returns Prometheus metrics for monitoring.

## 🎼 Music Prompting Guide

### Effective Prompting Strategies

#### 1. **Be Descriptive**
Use adjectives that describe mood, genre, and instrumentation:

✅ **Good Examples:**
- "Melancholic jazz with soft saxophone and gentle piano"
- "Energetic rock with electric guitars and powerful drums"
- "Peaceful ambient with ethereal pads and gentle wind chimes"
- "Funky disco with groovy basslines and rhythmic percussion"

❌ **Poor Examples:**
- "Music"
- "Song"
- "Beat"

#### 2. **Genre-Specific Prompts**

**Jazz:**
```
"Cool jazz with smooth saxophone, brushed drums, and walking bass"
"Bebop jazz with fast piano runs and complex drum patterns"
"Lounge jazz with soft vibraphone and gentle brushwork"
```

**Rock:**
```
"Classic rock with electric guitar riffs and driving drums"
"Progressive rock with complex time signatures and synthesizers"
"Blues rock with soulful guitar solos and steady rhythm section"
```

**Electronic:**
```
"Minimal techno with deep bass and repetitive synth patterns"
"Ambient electronic with atmospheric pads and gentle arpeggios"
"House music with four-on-the-floor beats and funky basslines"
```

**Classical:**
```
"Baroque classical with harpsichord and string ensemble"
"Romantic classical with emotional piano and orchestral strings"
"Minimalist classical with repetitive piano patterns"
```

**World Music:**
```
"Afrobeat with polyrhythmic drums and call-and-response vocals"
"Flamenco with passionate guitar and rhythmic handclaps"
"Indian classical with sitar and tabla rhythms"
```

### 3. **Iterative Prompting**
Instead of completely changing prompts, gradually modify elements:

**Start with:**
```
"Jazz with piano and drums"
```

**Then add:**
```
"Jazz with piano, drums, and walking bass"
```

**Then refine:**
```
"Cool jazz with soft piano, brushed drums, and walking bass"
```

## 🎛️ Music Parameters

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | required | - | Text description of the music |
| `duration_seconds` | integer | 25 | 1-300 | Length of generated music |
| `bpm` | integer | 110 | 60-200 | Beats per minute |
| `temperature` | float | 1.2 | 0.1-2.0 | Creativity/randomness |
| `density` | float | 0.8 | 0.1-1.0 | Musical complexity |
| `brightness` | float | 0.6 | 0.1-1.0 | Overall brightness |
| `scale` | string | "G_MAJOR_E_MINOR" | See below | Musical scale |
| `guidance` | float | 4.5 | 1.0-10.0 | Prompt adherence |

### Available Scales

| Scale Name | Description |
|------------|-------------|
| `G_MAJOR_E_MINOR` | G major / E minor |
| `C_MAJOR_A_MINOR` | C major / A minor |
| `F_MAJOR_D_MINOR` | F major / D minor |
| `B_FLAT_MAJOR_G_MINOR` | B♭ major / G minor |
| `E_FLAT_MAJOR_C_MINOR` | E♭ major / C minor |
| `A_FLAT_MAJOR_F_MINOR` | A♭ major / F minor |
| `D_FLAT_MAJOR_B_FLAT_MINOR` | D♭ major / B♭ minor |
| `G_FLAT_MAJOR_E_FLAT_MINOR` | G♭ major / E♭ minor |
| `B_MAJOR_A_FLAT_MINOR` | B major / A♭ minor |
| `E_MAJOR_D_FLAT_MINOR` | E major / D♭ minor |
| `A_MAJOR_G_FLAT_MINOR` | A major / G♭ minor |
| `D_MAJOR_B_MINOR` | D major / B minor |

### Parameter Guidelines

**Temperature (0.1-2.0):**
- **0.1-0.5**: Very predictable, repetitive patterns
- **0.5-1.0**: Balanced creativity and coherence
- **1.0-1.5**: More creative, varied patterns
- **1.5-2.0**: Highly creative, experimental

**Density (0.1-1.0):**
- **0.1-0.3**: Sparse, minimal arrangements
- **0.3-0.7**: Moderate complexity
- **0.7-1.0**: Dense, complex arrangements

**Brightness (0.1-1.0):**
- **0.1-0.3**: Dark, moody tones
- **0.3-0.7**: Balanced brightness
- **0.7-1.0**: Bright, cheerful tones

**Guidance (1.0-10.0):**
- **1.0-3.0**: Loose adherence to prompt
- **3.0-7.0**: Balanced adherence
- **7.0-10.0**: Strict adherence to prompt

## 🔧 Technical Details

### Audio Specifications
- **Format**: WAV (48kHz, 16-bit, stereo)
- **Sample Rate**: 48kHz
- **Channels**: 2 (stereo)
- **Bit Depth**: 16-bit PCM

### Production Features
- ✅ **10 API keys** with automatic rotation
- ✅ **Retry logic** for reliability (3 retries with 5s delay)
- ✅ **Rate limiting** (60/min, 1000/hour)
- ✅ **Structured logging** with JSON format
- ✅ **Prometheus metrics** for monitoring
- ✅ **Health checks** and status endpoints
- ✅ **Background task processing**
- ✅ **File cleanup** (7 days retention)

### Limitations
- **Instrumental only**: No vocals or lyrics
- **Safety filters**: Prompts are checked by safety filters
- **Watermarking**: Output audio is watermarked for identification
- **Real-time streaming**: Uses WebSocket connections for low-latency generation

## 📱 Client Integration Examples

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function generateMusic(prompt) {
    const API_URL = 'https://your-app-name.onrender.com';
    
    // 1. Start generation
    const response = await axios.post(`${API_URL}/generate-music`, {
        prompt: prompt,
        duration_seconds: 30,
        bpm: 120,
        temperature: 1.2
    });
    
    const { task_id } = response.data;
    
    // 2. Poll for completion
    let status;
    do {
        const statusResponse = await axios.get(`${API_URL}/music-status/${task_id}`);
        status = statusResponse.data;
        if (status.status === 'generating') {
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    } while (status.status === 'generating');
    
    // 3. Download music
    if (status.status === 'completed') {
        const musicResponse = await axios.get(`${API_URL}/download-music/${task_id}`, {
            responseType: 'stream'
        });
        return musicResponse.data;
    }
}
```

### Python
```python
import requests
import time

def generate_music(prompt, api_url="https://your-app-name.onrender.com"):
    # 1. Start generation
    response = requests.post(f"{api_url}/generate-music", json={
        'prompt': prompt,
        'duration_seconds': 30,
        'bpm': 120,
        'temperature': 1.2
    })
    
    task_id = response.json()['task_id']
    
    // 2. Poll for completion
    while True:
        status_response = requests.get(f"{api_url}/music-status/{task_id}")
        status = status_response.json()
        
        if status['status'] == 'completed':
            // 3. Download music
            music_response = requests.get(f"{api_url}/download-music/{task_id}")
            return music_response.content
        elif status['status'] == 'failed':
            raise Exception(f"Generation failed: {status.get('error')}")
        
        time.sleep(2)
```

### cURL Examples
```bash
# Generate music
curl -X POST "https://your-app-name.onrender.com/generate-music" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Afrobeat with smooth pianos and drums", "duration_seconds": 25}'

# Check status
curl "https://your-app-name.onrender.com/music-status/{task_id}"

# Download music
curl "https://your-app-name.onrender.com/download-music/{task_id}" --output music.wav

# Health check
curl "https://your-app-name.onrender.com/health"
```

## 📁 Project Structure

```
sofy-music-model/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── render.yaml           # Render deployment config
├── README.md            # This file
└── uploads/             # Generated music files (created automatically)
```

## 🔒 Security & Best Practices

### 1. **API Key Management**
- Never commit API keys to version control
- Use Render's environment variables for secrets
- Monitor API key usage via `/api-keys/stats`
- Rotate keys regularly

### 2. **Rate Limiting**
- Default: 60 requests per minute
- Adjust based on your needs in Render dashboard
- Monitor usage to avoid hitting limits

### 3. **Error Handling**
- API automatically retries failed requests
- Switches API keys on rate limits
- Provides detailed error messages

### 4. **Monitoring**
- Use `/health` for basic health checks
- Use `/metrics` for detailed monitoring
- Monitor logs in Render dashboard

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   curl https://your-app-name.onrender.com/api-keys/stats
   ```

2. **Rate Limiting**
   - Check current limits in response headers
   - Reduce request frequency if needed

3. **Generation Failures**
   - Check logs in Render dashboard
   - Verify API keys are valid
   - Ensure prompts follow guidelines

4. **Slow Response Times**
   - Music generation takes time (25-300 seconds)
   - Use status endpoint to track progress
   - Consider shorter durations for testing

## 📞 Support

- **Documentation**: Check `/docs` endpoint for interactive API docs
- **Health Check**: Use `/health` to verify service status
- **Metrics**: Use `/metrics` for monitoring
- **Logs**: Check Render dashboard for detailed logs

---

**Happy music making! 🎵** 