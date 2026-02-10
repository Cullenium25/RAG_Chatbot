import config
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def run_ingestion():
    # 1. Load documents
    print(f"Loading files from {config.DATA_DIR}...")
    loader = DirectoryLoader(config.DATA_DIR, glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # 2. Split into clinical-friendly chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} docs into {len(chunks)} chunks.")

    # 3. Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PATH,
        collection_name=config.COLLECTION_NAME
    )
    print("Success: Vector database updated.")

if __name__ == "__main__":
    run_ingestion()