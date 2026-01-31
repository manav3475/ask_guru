from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

import traceback
import config
from src.retrieval import retrieve_documents
from src.generate import generate_answer

app = FastAPI(title="Ask Guru", version="1.0")

# Qdrant client
client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
    timeout=120.0
)

# -------- Request / Response Models --------

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str

# -------- Health Check --------

@app.get("/")
def health():
    return {"status": "OK"}

# -------- Chat Endpoint --------

# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     try:
#         # Retrieve documents
#         docs = retrieve_documents(
#             client=client,
#             query=request.query,
#             collection_name=config.COLLECTION_NAME,
#             k=request.top_k
#         )

#         if not docs:
#             raise HTTPException(status_code=404, detail="No documents found")

#         # Generate answer
#         answer = generate_answer(
#             query=request.query,
#             retrieved_docs=docs
#         )

#         return ChatResponse(answer=answer)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print("📩 Query:", request.query)

        # ✅ ADD THIS HERE (early return)
        if request.query.lower().strip() in ["hi", "hello", "hey"]:
            return ChatResponse(answer="Hello! How can I help you today?")

        docs = retrieve_documents(
            client=client,
            query=request.query,
            collection_name=config.COLLECTION_NAME,
            k=request.top_k
        )

        if not docs:
            return ChatResponse(answer="❌ No relevant documents found.")

        answer = generate_answer(
            query=request.query,
            retrieved_docs=docs
        )

        return ChatResponse(answer=answer)

    except Exception as e:
        print("🔥 CHAT ENDPOINT ERROR")
        traceback.print_exc()

        return ChatResponse(
            answer="❌ Backend error. Check FastAPI logs."
        )
