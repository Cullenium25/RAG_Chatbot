import os
import config
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Initialize LLM and Embeddings
llm = ChatOpenAI(base_url=config.LLM_URL, api_key="lm-studio", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

# Connect to the existing Vector DB
vector_db = Chroma(
    persist_directory=config.CHROMA_PATH, 
    embedding_function=embeddings, 
    collection_name=config.COLLECTION_NAME
)


# Setup Multi-Query Retriever
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=config.MULTI_QUERY_PROMPT
)

smart_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K}), 
    llm=llm,
    prompt=QUERY_PROMPT
)

# Define Persona
system_prompt = (
    "You are the official AI Research Assistant for a Clinical Research Training Program. "
    "Use the following transcript context to answer questions accurately. "
    "If the answer isn't in the context, inform the user it is not available. "
    "\n\nCONTEXT:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Assemble Chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(smart_retriever, qa_chain)

def start_session():
    print("\n" + "="*50)
    print(" CLINICAL RESEARCH ASSISTANT ONLINE")
    print("="*50)

    while True:
        query = input("\nStudent Question: ").strip()
        if query.lower() in ["quit", "exit", "q"]: break
        if not query: continue

        try:
            response = rag_chain.invoke({"input": query})
            print(f"\nAI RESPONSE:\n{response['answer']}")
            
            # --- SOURCE VALIDATION ---
            print("\nSOURCES REFERENCED:")
            sources = {os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in response['context']}
            for i, src in enumerate(sources, 1):
                print(f"[{i}] {src}")
                
        except Exception as e:
            print(f"Error: {e}. Ensure LM Studio is running.")

if __name__ == "__main__":
    start_session()