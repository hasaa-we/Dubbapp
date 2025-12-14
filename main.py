import uuid
import subprocess
import os

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from openai import OpenAI
from elevenlabs import ElevenLabs

# ========================
# CONFIG (ENV VARS ONLY)
# ========================

POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not all([POE_ACCESS_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY]):
    raise RuntimeError("Missing required environment variables")

ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # default voice

# ========================
# CLIENTS
# ========================

openai_client = OpenAI(api_key=OPENAI_API_KEY)
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ========================
# APP
# ========================

app = FastAPI(title="Poe Video Dubbing API")

# ========================
# ENDPOINT
# ========================

@app.post("/poe-dub")
async def dub_video(
    video: UploadFile = File(...),
    target_language: str = Form(...),
    x_poe_access_key: str = Header(None),
):
    # 🔐 Poe authentication
    if x_poe_access_key != POE_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid.uuid4())

    input_video = f"{job_id}.mp4"
    audio_wav = f"{job_id}.wav"
    dubbed_wav = f"{job_id}_dub.wav"
    output_video = f"{job_id}_output.mp4"

    # Save uploaded video
    with open(input_video, "wb") as f:
        f.write(await video.read())

    # Extract audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_video, audio_wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Transcribe audio
    with open(audio_wav, "rb") as f:
        transcription = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
        )

    text = transcription.text

    # Translate text
    translation = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate the following text into {target_language}."},
            {"role": "user", "content": text},
        ],
    )

    translated_text = translation.choices[0].message.content

    # Generate dubbed voice
    audio_bytes = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=translated_text,
    )

    with open(dubbed_wav, "wb") as f:
        f.write(audio_bytes)

    # Merge dubbed audio back into video
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-i",
            dubbed_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-shortest",
            output_video,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return {
        "status": "done",
        "output_video": output_video,
        "target_language": target_language,
    }
