# ContinuEd RAG: Clinical Research AI Assistant
An intelligent Retrieval-Augmented Generation (RAG) system designed to assist researchers and students in navigating complex clinical research transcripts. This project leverages local LLM inference and advanced multi-query retrieval to provide precise, context-aware answers.

## Key Features
Multi-Query Retrieval: Implements LangChain's MultiQueryRetriever to generate 5 distinct variations of user questions. This overcomes the limitations of standard semantic search by capturing different perspectives of a query.

Source Attribution: Every response includes automated source validation, listing exactly which transcripts were referenced to ensure clinical reliability.

Privacy-First (Local LLM): Integrated with LM Studio and HuggingFace embeddings to ensure all data processing remains on your local Fedora machine.

Persistent Vector Store: Utilizes ChromaDB for fast, local vector storage and retrieval.

## Tech Stack
Framework: LangChain

Vector Database: ChromaDB

LLM Engine: LM Studio (Local Inference)

Embeddings: HuggingFace all-MiniLM-L6-v2

Environment: Python 3.x (Fedora Linux)

## Project Structure
``` text.
├── ingest.py            # Script to process documents and build vector DB
├── chat.py              # Main interactive RAG chat interface
├── config.py            # Centralized configuration and hyper-parameters
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── data/                # [Local Only] Folder for source transcripts (.txt, .pdf)
```
## Getting Started
### 1. Prerequisites
LM Studio: Download and run a model. Start the Local Server on http://localhost:1234.

Python: Ensure Python 3.10+ is installed on your Fedora system.

### 2. Installation & Setup
#### Clone the repository

```Bash
git clone https://github.com/YOUR_USERNAME/continued-rag-assistant.git
cd continued-rag-assistant
Setup Virtual Environment
```

#### Setup Python Environment by creating virtual environment for dependencies
```Bash
python3 -m venv .venv
source .venv/bin/activate
Install Dependencies
``` 
#### Install necessary libraries
```Bash
pip install -r requirements.txt
```
#### Configure LM Studio

Start LM Studio and load your model.

Ensure the "Local Server" is active on http://localhost:1234.

### 3. Usage
Prepare Data: Place your transcript files in the data/ folder.

Ingest Documents: Run python ingest.py.

Start Chatting: Run python chat.py.

## Disclaimer
This repository contains original source code for the RAG architecture. The course transcripts and proprietary clinical data used during development are not included in this repository to respect intellectual property and copyright. Users must provide their own data for ingestion.