import os
import re
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from models import SignUpRequest, LoginRequest, AuthResponse, User
from database import users_collection as user_collection
from utils.auth import hash_password, verify_password, create_token

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL   = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

app = FastAPI(title="MeetAI Auth API")


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

    # Call HuggingFace Inference API
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": cleaned}
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Summarization failed. Try again.")

    result = response.json()
    summary = result[0]["summary_text"].strip()

    return {"summary": summary}


@app.get("/")
def root():
    return {"status": "MeetAI API is running."}