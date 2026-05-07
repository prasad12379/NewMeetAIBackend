import os
import re
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from models import SignUpRequest, LoginRequest, AuthResponse, User
from database import users_collection as user_collection
from utils.auth import hash_password, verify_password, create_token

import tempfile
import shutil
from pathlib import Path
from groq import Groq
from fastapi.responses import Response

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # add at top with other clients
HF_API_URL   = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

app = FastAPI(title="MeetAI Auth API")


# ══════════════════════════════════════════════════════════
# TEXT CLEANER
# ══════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = re.sub(
        r"\b(uh|um|basically|you know|kind of|i mean|hey|alright|thanks everyone)\b",
        "", text, flags=re.IGNORECASE
    )
    flat = " ".join(
        re.sub(r"^\w+:\s*", "", line).strip()
        for line in text.splitlines() if line.strip()
    )
    return re.sub(r" {2,}", " ", flat).strip()


# ══════════════════════════════════════════════════════════
# CHUNKER — splits large text into 900-char chunks
# ══════════════════════════════════════════════════════════
def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    sentences = text.split(". ")
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks


# ══════════════════════════════════════════════════════════
# HF SUMMARIZER — handles large text via chunking
# ══════════════════════════════════════════════════════════
async def hf_summarize(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "parameters": {
            "max_length": 130,
            "min_length": 40,
            "do_sample": False
        }
    }

    chunks = chunk_text(text)
    summaries = []

    async with httpx.AsyncClient(timeout=60) as client:
        for chunk in chunks:
            payload["inputs"] = chunk
            response = await client.post(HF_API_URL, headers=headers, json=payload)

            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="Model is loading, retry in 20 seconds.")
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"HF API error {response.status_code}: {response.text}")

            result = response.json()
            if not result or not isinstance(result, list) or "summary_text" not in result[0]:
                raise HTTPException(status_code=500, detail=f"Unexpected HF response: {result}")

            summaries.append(result[0]["summary_text"].strip())

    # If multiple chunks, do a final summarization pass on combined summaries
    combined = " ".join(summaries)
    if len(chunks) > 1 and len(combined) > 200:
        payload["inputs"] = combined[:900]
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                final = response.json()
                if final and isinstance(final, list) and "summary_text" in final[0]:
                    return final[0]["summary_text"].strip()

    return combined


# ══════════════════════════════════════════════════════════
# /signup
# ══════════════════════════════════════════════════════════
@app.post("/signup", response_model=AuthResponse)
async def signup(data: SignUpRequest):
    existing = await user_collection.find_one({"email": data.email})
    if existing:
        return AuthResponse(success=False, message="Email already registered.")

    existing_username = await user_collection.find_one({"username": data.username})
    if existing_username:
        return AuthResponse(success=False, message="Username already taken.")

    new_user = {
        "email":    data.email,
        "username": data.username,
        "password": hash_password(data.password)
    }
    result  = await user_collection.insert_one(new_user)
    user_id = str(result.inserted_id)
    token   = create_token(user_id)

    return AuthResponse(
        success=True,
        message="Account created successfully.",
        user=User(id=user_id, email=data.email, username=data.username),
        token=token
    )


# ══════════════════════════════════════════════════════════
# /login
# ══════════════════════════════════════════════════════════
@app.post("/login", response_model=AuthResponse)
async def login(data: LoginRequest):
    user = await user_collection.find_one({"email": data.email})
    if not user:
        return AuthResponse(success=False, message="Email not found.")

    if not verify_password(data.password, user["password"]):
        return AuthResponse(success=False, message="Incorrect password.")

    user_id = str(user["_id"])
    token   = create_token(user_id)

    return AuthResponse(
        success=True,
        message="Login successful.",
        user=User(id=user_id, email=user["email"], username=user["username"]),
        token=token
    )


# ══════════════════════════════════════════════════════════
# /summarize  →  returns summary JSON
# ══════════════════════════════════════════════════════════
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted.")

    raw = await file.read()

    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    if not text:
        raise HTTPException(status_code=400, detail="File is empty.")

    cleaned = clean_text(text)
    summary = await hf_summarize(cleaned)

    return {"summary": summary}


# ══════════════════════════════════════════════════════════
# /debug-token  →  remove after confirming token works
# ══════════════════════════════════════════════════════════
@app.get("/debug-token")
def debug_token():
    return {
        "token_set": HF_API_TOKEN is not None,
        "token_preview": HF_API_TOKEN[:8] if HF_API_TOKEN else "MISSING"
    }


# ══════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MeetAI API is running."}


# ══════════════════════════════════════════════════════════
# /transcribe  →  accepts audio, returns .txt file
# ══════════════════════════════════════════════════════════
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm", ".flac"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Groq has a 25MB file size limit
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 25MB.")

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename, contents),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    txt_filename = Path(file.filename).stem + "_transcript.txt"
    return Response(
        content=transcription,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={txt_filename}"}
    )