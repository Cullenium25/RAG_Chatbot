import os

# Paths
CHROMA_PATH = "./vector_db"
DATA_DIR = "./data/"
COLLECTION_NAME = "Test_Course_Name"

# Model Config
EMBED_MODEL = "all-MiniLM-L6-v2"  # Or "BAAI/bge-m3" for higher accuracy
LLM_URL = "http://localhost:1234/v1"

# Retrieval Hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 6

# Multi-Query Configuration
NUM_VARIATIONS = 5 # Set this to whatever number you want
MULTI_QUERY_PROMPT = f"""You are an AI language model assistant. 
Your task is to generate {NUM_VARIATIONS} different versions of the given user question 
to retrieve relevant documents from a vector database. By generating multiple perspectives 
on the user query, your goal is to help the user overcome some of the limitations 
of the distance-based similarity search. 

Original question: {{question}}"""