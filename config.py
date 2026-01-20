import os
from dotenv import load_dotenv

load_dotenv(".env")

# Secrets
HUGGINGFACEHUB_ACCESS_TOKEN = os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# Directories
DATA_DIR = "./data/*.pdf"

# Qdrant Params
COLLECTION_NAME = "Ask_Guru"

# Hyperparameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 215
