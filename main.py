import os
import re
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from models import SignUpRequest, LoginRequest, AuthResponse, User
from database import users_collection as user_collection
from utils.auth import hash_password, verify_password, create_token
import asyncio
from pathlib import Path
from groq import Groq
from fastapi.responses import Response
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, List
import smtplib
import base64

import resend

load_dotenv()

SMTP_EMAIL    = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
HF_API_TOKEN  = os.getenv("HF_API_TOKEN")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
MONGO_URL     = os.getenv("MONGO_URL")

groq_client = Groq(api_key=GROQ_API_KEY)
HF_API_URL  = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

app = FastAPI(title="MeetAI Auth API")

# ══════════════════════════════════════════════════════════
# MongoDB — lazy init on startup event
# ══════════════════════════════════════════════════════════
meetings_collection = None

@app.on_event("startup")
async def startup_db():
    global meetings_collection
    if not MONGO_URL:
        raise RuntimeError("MONGO_URL environment variable is not set.")
    try:
        meetings_client     = MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
        meetings_db         = meetings_client["MeetAIdb"]
        meetings_collection = meetings_db["Meetings"]
        meetings_collection.create_index("room_code", unique=True)
        print("✅ MongoDB connected successfully.")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise RuntimeError(f"MongoDB connection failed: {e}")

# ══════════════════════════════════════════════════════════
# Pydantic Models
# ══════════════════════════════════════════════════════════
class MeetingRequest(BaseModel):
    room_code: str
    email:     str
    is_admin:  bool

class SendSummaryRequest(BaseModel):
    room_code:   str
    duration:    str
    members:     List[str]
    pdf_base64:  str
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
                "role":    "system",
                "content": "Merge these partial meeting summaries into one complete, detailed summary. Do not lose any detail."
            },
            {
                "role":    "user",
                "content": f"Merge:\n\n{combined}"
            }
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return final.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════
# SMTP email sender — runs in thread pool (non-blocking)
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# RESEND email sender — runs in thread pool (non-blocking)
# ══════════════════════════════════════════════════════════
def send_emails_sync(
    members:      List[str],
    pdf_bytes:    bytes,
    pdf_filename: str,
    room_code:    str,
    duration:     str,
    admin_email:  str
) -> dict:
    """
    Sends PDF to all members via Resend API.
    Runs in asyncio.to_thread() to avoid blocking FastAPI.
    """
    failed  = []
    success = []

    # Encode PDF to base64 for Resend attachment
    pdf_base64_str = base64.b64encode(pdf_bytes).decode("utf-8")

    for member_email in members:
        try:
            body_html = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #0D0D1A; padding: 30px; border-radius: 12px;">
                    <h2 style="color: #2ACFD9; margin-bottom: 8px;">
                        MeetAI — Meeting Summary
                    </h2>
                    <hr style="border-color: #2ACFD9; margin-bottom: 20px;">

                    <p style="color: #CCCCCC;">Hi,</p>
                    <p style="color: #CCCCCC;">
                        Please find attached the AI-generated summary
                        for your recent MeetAI meeting.
                    </p>

                    <div style="background: #1A1A2E; border-radius: 8px;
                                padding: 16px; margin: 20px 0;">
                        <p style="color: #888888; margin: 0 0 8px 0;
                                  font-size: 12px; letter-spacing: 1px;">
                            MEETING DETAILS
                        </p>
                        <p style="color: #2ACFD9; margin: 4px 0;">
                            🏠 Room Code: <strong>{room_code}</strong>
                        </p>
                        <p style="color: #FFC107; margin: 4px 0;">
                            ⏱ Duration: <strong>{duration}</strong>
                        </p>
                        <p style="color: #A855F7; margin: 4px 0;">
                            👤 Admin: <strong>{admin_email}</strong>
                        </p>
                    </div>

                    <p style="color: #CCCCCC;">
                        The full meeting summary is attached as a PDF.
                    </p>

                    <hr style="border-color: #333333; margin-top: 20px;">
                    <p style="color: #555555; font-size: 12px;">
                        Sent by MeetAI • Powered by AI
                    </p>
                </div>
            </div>
            """

            params = {
                "from":    "MeetAI <onboarding@resend.dev>",  # use this until you verify domain
                "to":      [member_email],
                "subject": f"MeetAI — Meeting Summary [{room_code}]",
                "html":    body_html,
                "attachments": [
                    {
                        "filename": pdf_filename,
                        "content":  pdf_base64_str,
                    }
                ]
            }

            response = resend.Emails.send(params)
            print(f"✅ Email sent to {member_email} | ID: {response.get('id', 'N/A')}")
            success.append(member_email)

        except Exception as e:
            print(f"❌ Failed to send to {member_email}: {e}")
            failed.append({"email": member_email, "error": str(e)})

    return {
        "success": success,
        "failed":  failed
    }

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
# ══════════════════════════════════════════════════════════
@app.post("/meetings")
async def handle_meeting(data: MeetingRequest):
    if meetings_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    room_code = data.room_code.strip().upper()
    email     = data.email.strip().lower()

    if data.is_admin:
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
        meeting_doc = {
            "room_code":   room_code,
            "admin_email": email,
            "members":     [email],
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }
        try:
            meetings_collection.insert_one(meeting_doc)
        except DuplicateKeyError:
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

    else:
        existing = meetings_collection.find_one({"room_code": room_code})
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "status":  "error",
                    "message": f"Room '{room_code}' does not exist.",
                    "code":    "ROOM_NOT_FOUND"
                }
            )
        meetings_collection.update_one(
            {"room_code": room_code},
            {"$addToSet": {"members": email}}
        )
        updated = meetings_collection.find_one({"room_code": room_code}, {"_id": 0})
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
# ══════════════════════════════════════════════════════════
@app.get("/meetings/{email}")
async def get_meetings(email: str):
    if meetings_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    email    = email.strip().lower()
    meetings = list(
        meetings_collection.find(
            {"$or": [{"admin_email": email}, {"members": email}]},
            {"_id": 0}
        ).sort("created_at", -1)
    )
    return {
        "status":   "success",
        "email":    email,
        "count":    len(meetings),
        "meetings": meetings
    }

# ══════════════════════════════════════════════════════════
# /meetings/{room_code}/members  →  GET
# ══════════════════════════════════════════════════════════
@app.get("/meetings/{room_code}/members")
async def get_meeting_members(room_code: str):
    if meetings_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    room_code = room_code.strip().upper()
    meeting   = meetings_collection.find_one(
        {"room_code": room_code},
        {"_id": 0, "members": 1, "admin_email": 1}
    )
    if not meeting:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting '{room_code}' not found."
        )
    return {
        "room_code":   room_code,
        "admin_email": meeting.get("admin_email", ""),
        "members":     meeting.get("members", [])
    }

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
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

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

# ══════════════════════════════════════════════════════════
# /send-summary  →  sends PDF to all meeting members
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# /send-summary  →  sends PDF to all meeting members
# ══════════════════════════════════════════════════════════
@app.post("/send-summary")
async def send_summary(data: SendSummaryRequest):

    # ── Validate Resend API key ───────────────────────────────────
    if not RESEND_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="RESEND_API_KEY not configured. Add it to environment variables."
        )

    if not data.members:
        raise HTTPException(status_code=400, detail="No members to send to.")

    # ── Decode PDF ────────────────────────────────────────────────
    try:
        pdf_bytes = base64.b64decode(data.pdf_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid PDF base64 data: {str(e)}"
        )

    if len(pdf_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="PDF data appears to be empty or corrupted."
        )

    pdf_filename = f"Summary_{data.room_code}.pdf"

    print(f"📧 Sending to {len(data.members)} member(s): {data.members}")
    print(f"📄 PDF size: {len(pdf_bytes)} bytes")

    # ── Send in thread pool (non-blocking) ───────────────────────
    try:
        result = await asyncio.to_thread(
            send_emails_sync,
            data.members,
            pdf_bytes,
            pdf_filename,
            data.room_code,
            data.duration,
            data.admin_email
        )
    except Exception as e:
        print(f"❌ Email sending error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    success = result["success"]
    failed  = result["failed"]

    return {
        "status":       "success" if not failed else "partial",
        "sent_to":      success,
        "failed":       failed,
        "total_sent":   len(success),
        "total_failed": len(failed)
    }
# ══════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status":       "MeetAI API is running.",
        "smtp_email":   SMTP_EMAIL or "NOT SET",
        "mongo_url":    "SET" if MONGO_URL else "NOT SET",
        "groq_key":     "SET" if GROQ_API_KEY else "NOT SET"
    }