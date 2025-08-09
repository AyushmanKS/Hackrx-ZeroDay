import os
import requests
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# LangChain components for the advanced solution
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, HydeRetriever

# --- Configuration ---
PDF_STORAGE_PATH = "policy.pdf"
CHAIN_CACHE: Dict[str, RetrievalQA] = {}

# --- API Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Core Logic ---

PROMPT_V_EXPERT = """
You are an AI expert specializing in insurance policy analysis. Your task is to provide clear and accurate answers based *exclusively* on the provided text excerpts from a policy document.
Use the following context to answer the user's question. Synthesize the information from all provided chunks to form a complete answer.
If the information is present, provide a direct and factual answer.
If the context does not contain the answer, state: "A clear answer to this question could not be found in the provided document excerpts."

Context:
{context}

Question:
{question}

Factual and Precise Answer:
"""

def create_and_cache_qa_chain(doc_url: str):
    """
    Creates the definitive high-accuracy/high-performance QA chain and caches it.
    Uses Parent Document Retriever for context and HyDE for retrieval accuracy.
    """
    if doc_url in CHAIN_CACHE:
        print(f"[INFO] Using cached QA chain for document: {doc_url}")
        return CHAIN_CACHE[doc_url]

    print(f"[INFO] New document. Creating definitive QA chain for: {doc_url}")
    
    response = requests.get(doc_url)
    response.raise_for_status()
    with open(PDF_STORAGE_PATH, 'wb') as f:
        f.write(response.content)
    
    loader = PyPDFLoader(PDF_STORAGE_PATH)
    docs = loader.load()

    # --- ADVANCED RETRIEVAL SETUP ---

    # 1. Parent Document Retriever Setup
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    vectorstore = FAISS.from_documents(docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")) # Temporary store for setup
    store = InMemoryStore()
    
    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_document_retriever.add_documents(docs)

    # 2. HyDE (Hypothetical Document Embeddings) Retriever Setup
    llm_for_hyde = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    embeddings_for_hyde = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    hyde_retriever = HydeRetriever(
        retriever=parent_document_retriever,
        llm=llm_for_hyde,
        prompt_key="query" # Pass the question to the LLM
    )

    # 3. Final QA Chain Assembly
    llm_for_qa = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    prompt = PromptTemplate(template=PROMPT_V_EXPERT, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_qa,
        chain_type="stuff",
        retriever=hyde_retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False
    )
    
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] Definitive QA chain for {doc_url} created and cached.")
    
    if os.path.exists(PDF_STORAGE_PATH):
        os.remove(PDF_STORAGE_PATH)
        
    return qa_chain

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(req: HackRxRequest):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")
    
    try:
        qa_chain = create_and_cache_qa_chain(req.documents)
        
        # TECHNIQUE 1: Batch processing for maximum speed
        # Create a list of inputs for the batch call
        batch_inputs = [{"query": q} for q in req.questions]
        results = await qa_chain.abatch(batch_inputs)
        
        answers = [res["result"] for res in results]
            
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running with advanced accuracy and performance enhancements."}