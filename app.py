import os
import shutil
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

import matchering as mg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feelize_mastering")

app = FastAPI(title="Feelize Studio Mastering Engine", version="2.0.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")
os.makedirs(REFERENCES_DIR, exist_ok=True)

PRESET_FILES = {
    "commercial_radio": "commercial_radio.wav",
    "club_urban": "club_urban.wav",
    "tape_warmth": "tape_warmth.wav",
}

def remove_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {path}: {e}")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Feelize Studio 64-Bit Mastering Microservice",
        "engine": "Matchering 2.0.6",
        "precision": "64-bit float FFT convolution",
        "limiter": "Hyrax multi-stage lookahead brickwall (-0.1 dBTP)",
    }

@app.get("/health")
def health_check():
    available_presets = []
    for preset, filename in PRESET_FILES.items():
        ref_path = os.path.join(REFERENCES_DIR, filename)
        if os.path.exists(ref_path):
            available_presets.append(preset)
            
    return {
        "status": "ready",
        "available_presets": available_presets,
        "default_preset": "commercial_radio"
    }

@app.post("/master")
async def master_audio(
    file: UploadFile = File(...),
    preset: Optional[str] = Form("commercial_radio")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    preset_name = preset.lower() if preset else "commercial_radio"
    if preset_name not in PRESET_FILES:
        preset_name = "commercial_radio"

    ref_filename = PRESET_FILES[preset_name]
    ref_path = os.path.join(REFERENCES_DIR, ref_filename)

    # Fallback to commercial_radio if preset file does not exist
    if not os.path.exists(ref_path):
        ref_path = os.path.join(REFERENCES_DIR, "commercial_radio.wav")

    if not os.path.exists(ref_path):
        raise HTTPException(
            status_code=500,
            detail=f"Mastering reference profile not found on server: {ref_filename}"
        )

    # Create temporary paths
    file_ext = os.path.splitext(file.filename)[1].lower()
    if not file_ext:
        file_ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_in:
        temp_input_path = tmp_in.name
        shutil.copyfileobj(file.file, tmp_in)

    temp_output_path = temp_input_path + "_mastered.wav"

    try:
        logger.info(f"Processing audio: {file.filename} with preset: {preset_name}")
        
        # Configure Matchering for professional streaming standard
        config = mg.Config()
        config.max_length = 900  # Up to 15 minutes
        config.allow_equality = True

        # Process with 64-bit precision and 24-bit PCM WAV output
        mg.process(
            target=temp_input_path,
            reference=ref_path,
            results=[mg.pcm24(temp_output_path)],
            config=config
        )

        if not os.path.exists(temp_output_path):
            raise RuntimeError("Matchering finished but output file was not created.")

        logger.info(f"Mastering complete for {file.filename}. Output size: {os.path.getsize(temp_output_path)} bytes")

        # Cleanup input file immediately
        remove_temp_file(temp_input_path)

        # Return file and cleanup output after response is sent
        return FileResponse(
            temp_output_path,
            media_type="audio/wav",
            filename=f"mastered_{file.filename}.wav",
            background=BackgroundTask(remove_temp_file, temp_output_path)
        )

    except Exception as e:
        logger.error(f"Error during mastering: {e}", exc_info=True)
        remove_temp_file(temp_input_path)
        remove_temp_file(temp_output_path)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Mastering process failed: {str(e)}"}
        )
