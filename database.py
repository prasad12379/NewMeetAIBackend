from motor.motor_asyncio import AsyncIOMotorClient
import os


MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = "MeetAIdb"

client = AsyncIOMotorClient(MONGO_URL, tlsAllowInvalidCertificates=True)
db = client[DATABASE_NAME]
 
# Collections
users_collection = db["users"]

