import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document # <--- Added this

def run_ingestion():
    # 1. Load the new .md files
    print(f"Loading Markdown files from {config.DATA_DIR}...")
    loader = DirectoryLoader(config.DATA_DIR, glob="./*.md", loader_cls=TextLoader)
    documents = loader.load()

    # 2. SMART CHUNKING
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    for doc in documents:
        # Get the original source (e.g., './data/transcript_1.md')
        original_source = doc.metadata.get("source", "Unknown")
        
        # Split text into segments based on headers
        header_splits = header_splitter.split_text(doc.page_content)
        
        doc_splits = []
        for chunk in header_splits:
            # --- THE FIX: Merge header metadata with original file metadata ---
            combined_metadata = chunk.metadata.copy()
            combined_metadata["source"] = original_source
            
            doc_splits.append(
                Document(page_content=chunk.page_content, metadata=combined_metadata)
            )
        
        # Sub-split these documents (they now carry the correct source!)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        final_chunks.extend(text_splitter.split_documents(doc_splits))

    print(f"Prepared {len(final_chunks)} chunks. Creating Vector DB...")

    # 3. Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    
    vector_db = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PATH,
        collection_name=config.COLLECTION_NAME
    )
    
    # 4. Force Persistence (Required for some local versions)
    vector_db.persist() 
    print(f"✅ Success: Database created at {config.CHROMA_PATH}")

if __name__ == "__main__":
    run_ingestion()