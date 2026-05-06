from fastapi import FastAPI
from models import SignUpRequest, LoginRequest, AuthResponse, User
from database import users_collection as user_collection
from utils.auth import hash_password, verify_password, create_token

app = FastAPI(title="MeetAI Auth API")


# ══════════════════════════════════════════════════════════
# /signup
# ══════════════════════════════════════════════════════════
@app.post("/signup", response_model=AuthResponse)
async def signup(data: SignUpRequest):

    # Check if email already exists
    existing = await user_collection.find_one({"email": data.email})
    if existing:
        return AuthResponse(success=False, message="Email already registered.")

    # Check if username already exists
    existing_username = await user_collection.find_one({"username": data.username})
    if existing_username:
        return AuthResponse(success=False, message="Username already taken.")

    # Hash password and save to MongoDB
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

    # Find user by email
    user = await user_collection.find_one({"email": data.email})
    if not user:
        return AuthResponse(success=False, message="Email not found.")

    # Verify password
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
# Health check
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MeetAI API is running."}