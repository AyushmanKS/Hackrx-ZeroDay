import os
import requests
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# LangChain components - We need more tools now
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration ---
HACKRX_TOKEN = "ed219fa46cac4026e20b9562bdfeecd13373a1e0a7529dee5b196421a3d97d4d"
PDF_STORAGE_PATH = "policy.pdf"
# This is our simple in-memory cache for processed QA chains
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

# NEW: A powerful, specific prompt template to improve accuracy
PROMPT_TEMPLATE = """
You are a highly intelligent AI assistant for answering questions about a policy document.
You are given several chunks of text from the document as context.
Answer the user's question based *ONLY* on the provided context.
Do not use any outside knowledge or make assumptions.
If the answer is not present in the context, explicitly state: "The document does not provide this information."

Context:
{context}

Question:
{question}

Answer:
"""

def create_and_cache_qa_chain(doc_url: str):
    """
    Creates a high-performance QA chain and caches it.
    Uses MultiQueryRetriever and a strict prompt for accuracy.
    """
    # Check if we have already processed this document
    if doc_url in CHAIN_CACHE:
        print(f"[INFO] Using cached QA chain for document: {doc_url}")
        return CHAIN_CACHE[doc_url]

    print(f"[INFO] New document received. Creating new QA chain for: {doc_url}")
    
    # 1. Download and Load
    response = requests.get(doc_url)
    response.raise_for_status()
    with open(PDF_STORAGE_PATH, 'wb') as f:
        f.write(response.content)
    
    loader = PyPDFLoader(PDF_STORAGE_PATH)
    documents = loader.load()

    # 2. Split and Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(texts, embeddings)

    # 3. Create the Advanced Retriever and QA Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # UPGRADE 1: Use MultiQueryRetriever for better context
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=llm
    )
    
    # UPGRADE 2: Use our strict prompt template
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_from_llm,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False
    )
    
    # Store the newly created chain in our cache
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] QA chain for {doc_url} created and cached.")
    
    # Clean up the downloaded file
    if os.path.exists(PDF_STORAGE_PATH):
        os.remove(PDF_STORAGE_PATH)
        
    return qa_chain

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(req: HackRxRequest):
    # The Bearer token is not a dependency anymore as it's not used. 
    # The platform may still send it, but we don't need to validate it if not required by the logic.
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")
    
    try:
        qa_chain = create_and_cache_qa_chain(req.documents)
        
        # We can run these concurrently as the heavy lifting is already done.
        tasks = [qa_chain.ainvoke({"query": q}) for q in req.questions]
        results = await asyncio.gather(*tasks)
        answers = [res["result"] for res in results]
        
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running and ready for Level 2."}