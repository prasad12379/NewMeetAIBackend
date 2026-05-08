import os
import re
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from models import SignUpRequest, LoginRequest, AuthResponse, User
from database import users_collection as user_collection
from utils.auth import hash_password, verify_password, create_token
import asyncio
import tempfile
import shutil
from pathlib import Path
from groq import Groq
from fastapi.responses import Response
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_API_URL   = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

app = FastAPI(title="MeetAI Auth API")

# ══════════════════════════════════════════════════════════
# MongoDB — Meetings Collection
# ══════════════════════════════════════════════════════════
MONGO_URI = os.getenv("MONGO_URL")
meetings_client     = MongoClient(MONGO_URI)
meetings_db         = meetings_client["meetai"]
meetings_collection = meetings_db["Meetings"]

# Unique index on room_code — prevents duplicate rooms at DB level
meetings_collection.create_index("room_code", unique=True)

# ══════════════════════════════════════════════════════════
# Pydantic Model — /meetings
# ══════════════════════════════════════════════════════════
class MeetingRequest(BaseModel):
    room_code: str
    email:     str
    is_admin:  bool

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
# CHUNKER
# ══════════════════════════════════════════════════════════
def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
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
# HF SUMMARIZER
# ══════════════════════════════════════════════════════════
async def hf_summarize_chunk(
    client: httpx.AsyncClient,
    text: str,
    max_length: int = 200,
    min_length: int = 60
) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type":  "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample":  False
        }
    }

    response = await client.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code == 503:
        raise HTTPException(
            status_code=503,
            detail="Model is loading, retry in 20 seconds."
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"HF API error {response.status_code}: {response.text}"
        )

    result = response.json()
    if not result or not isinstance(result, list) or "summary_text" not in result[0]:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected HF response: {result}"
        )

    return result[0]["summary_text"].strip()

# ══════════════════════════════════════════════════════════
# GROQ SUMMARIZER
# ══════════════════════════════════════════════════════════
async def groq_summarize(text: str) -> str:
    MAX_CHARS = 12000
    chunks_to_summarize = (
        [text] if len(text) <= MAX_CHARS
        else chunk_text(text, max_chars=MAX_CHARS)
    )
    chunk_summaries = []

    for chunk in chunks_to_summarize:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting summarizer. "
                        "Produce a detailed structured summary covering: "
                        "1) Key discussion points, "
                        "2) Decisions made, "
                        "3) Problems or challenges identified, "
                        "4) Action items and who is responsible, "
                        "5) Next steps and deadlines. "
                        "Be thorough. Do not skip any department or topic."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this meeting transcript:\n\n{chunk}"
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        chunk_summaries.append(response.choices[0].message.content.strip())

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n".join(
        f"[Part {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    final = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Merge these partial meeting summaries into one "
                    "complete, detailed summary. Do not lose any detail."
                )
            },
            {
                "role": "user",
                "content": f"Merge:\n\n{combined}"
            }
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return final.choices[0].message.content.strip()

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
# /meetings  →  POST
# Handles both admin creating and member joining
# ══════════════════════════════════════════════════════════
@app.post("/meetings")
async def handle_meeting(data: MeetingRequest):
    """
    Single endpoint for meeting management:
      - is_admin = true  → Admin creates a new meeting
      - is_admin = false → Member joins existing meeting
    """

    room_code = data.room_code.strip().upper()
    email     = data.email.strip().lower()

    # ── CASE 1: Admin creating a new meeting ──────────────────────
    if data.is_admin:

        # Check if room already exists
        existing = meetings_collection.find_one({"room_code": room_code})
        if existing:
            raise HTTPException(
                status_code=409,
                detail={
                    "status":  "error",
                    "message": f"Room '{room_code}' already exists.",
                    "code":    "ROOM_ALREADY_EXISTS"
                }
            )

        # Build meeting document
        meeting_doc = {
            "room_code":   room_code,
            "admin_email": email,
            "members":     [email],   # admin is first member
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }

        try:
            meetings_collection.insert_one(meeting_doc)
        except DuplicateKeyError:
            # Race condition safety
            raise HTTPException(
                status_code=409,
                detail={
                    "status":  "error",
                    "message": f"Room '{room_code}' was just created by another request.",
                    "code":    "ROOM_ALREADY_EXISTS"
                }
            )

        return {
            "status":     "success",
            "message":    f"Meeting '{room_code}' created successfully.",
            "room_code":  room_code,
            "admin":      email,
            "members":    [email],
            "created_at": meeting_doc["created_at"]
        }

    # ── CASE 2: Member joining existing meeting ───────────────────
    else:

        # Check room exists
        existing = meetings_collection.find_one({"room_code": room_code})
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "status":  "error",
                    "message": f"Room '{room_code}' does not exist. "
                               f"Please check the room code.",
                    "code":    "ROOM_NOT_FOUND"
                }
            )

        # $addToSet prevents duplicate emails automatically
        meetings_collection.update_one(
            {"room_code": room_code},
            {"$addToSet": {"members": email}}
        )

        # Return updated document
        updated = meetings_collection.find_one(
            {"room_code": room_code},
            {"_id": 0}
        )

        return {
            "status":     "success",
            "message":    f"'{email}' joined meeting '{room_code}' successfully.",
            "room_code":  room_code,
            "admin":      updated["admin_email"],
            "members":    updated["members"],
            "created_at": updated["created_at"]
        }

# ══════════════════════════════════════════════════════════
# /meetings/{email}  →  GET
# Get all meetings for a user by email
# ══════════════════════════════════════════════════════════
@app.get("/meetings/{email}")
async def get_meetings(email: str):
    """
    Returns all meetings where this email
    is either the admin or a member.
    """
    email = email.strip().lower()

    meetings = list(
        meetings_collection.find(
            {"$or": [
                {"admin_email": email},
                {"members":     email}
            ]},
            {"_id": 0}
        ).sort("created_at", -1)  # newest first
    )

    return {
        "status":   "success",
        "email":    email,
        "count":    len(meetings),
        "meetings": meetings
    }

# ══════════════════════════════════════════════════════════
# /summarize  →  returns summary JSON
# ══════════════════════════════════════════════════════════
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are accepted."
        )

    raw = await file.read()

    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded."
        )

    if not text:
        raise HTTPException(status_code=400, detail="File is empty.")

    cleaned = clean_text(text)
    summary = await groq_summarize(cleaned)

    return {"summary": summary}

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
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Max size is 25MB."
        )

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename, contents),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

    txt_filename = Path(file.filename).stem + "_transcript.txt"
    return Response(
        content=transcription,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={txt_filename}"}
    )

# ══════════════════════════════════════════════════════════
# /debug-token
# ══════════════════════════════════════════════════════════
@app.get("/debug-token")
def debug_token():
    return {
        "token_set":     HF_API_TOKEN is not None,
        "token_preview": HF_API_TOKEN[:8] if HF_API_TOKEN else "MISSING"
    }

# ══════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MeetAI API is running."}