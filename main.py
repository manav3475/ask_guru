import os
import sys
import config

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv



load_dotenv()


app = FastAPI(title="ASK GURU", version="1.0")

@app.get("/")
def root():
    return {"message": "ASK GURU is up and running!"}


@app.get("/chat")
def root_endpoint():
    return {"message": "This is chat endpoint"}


# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("Explain the theory of relativity in simple terms.")

# print(result.content)
