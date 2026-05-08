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
import smtplib
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pydantic import BaseModel
from typing import List

load_dotenv()

SMTP_EMAIL    = os.getenv("SMTP_EMAIL")     # your gmail
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") 
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_API_URL   = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

app = FastAPI(title="MeetAI Auth API")

# ══════════════════════════════════════════════════════════
# MongoDB — Meetings Collection
# ══════════════════════════════════════════════════════════
MONGO_URI = os.getenv("MONGO_URL")
meetings_client     = MongoClient(MONGO_URI)
meetings_db         = meetings_client["MeetAIdb"]
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


class SendSummaryRequest(BaseModel):
    room_code:   str
    duration:    str
    members:     List[str]        # list of member emails
    pdf_base64:  str              # PDF file encoded as base64 string
    admin_email: str

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
# /send-summary  →  sends PDF to all meeting members
# ══════════════════════════════════════════════════════════
@app.post("/send-summary")
async def send_summary(data: SendSummaryRequest):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Email credentials not configured on server."
        )

    if not data.members:
        raise HTTPException(
            status_code=400,
            detail="No members to send to."
        )

    # Decode base64 PDF
    try:
        pdf_bytes = base64.b64decode(data.pdf_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid PDF data: {str(e)}"
        )

    pdf_filename = f"Summary_{data.room_code}.pdf"
    failed       = []
    success      = []

    try:
        # Connect to Gmail SMTP once, send to all members
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)

        for member_email in data.members:
            try:
                msg = MIMEMultipart()
                msg["From"]    = SMTP_EMAIL
                msg["To"]      = member_email
                msg["Subject"] = f"MeetAI — Meeting Summary [{data.room_code}]"

                # Email body
                body = f"""
Hi,

Please find attached the AI-generated summary for your recent MeetAI meeting.

Meeting Details:
  • Room Code : {data.room_code}
  • Duration  : {data.duration}
  • Admin     : {data.admin_email}

The full meeting summary is attached as a PDF.

Best regards,
MeetAI Team
                """.strip()

                msg.attach(MIMEText(body, "plain"))

                # Attach PDF
                part = MIMEBase("application", "octet-stream")
                part.set_payload(pdf_bytes)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={pdf_filename}"
                )
                msg.attach(part)

                server.sendmail(SMTP_EMAIL, member_email, msg.as_string())
                success.append(member_email)

            except Exception as e:
                failed.append({"email": member_email, "error": str(e)})

        server.quit()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SMTP connection failed: {str(e)}"
        )

    return {
        "status":        "success" if not failed else "partial",
        "sent_to":       success,
        "failed":        failed,
        "total_sent":    len(success),
        "total_failed":  len(failed)
    }


# ══════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MeetAI API is running."}