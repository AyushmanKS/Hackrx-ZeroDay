import os
import requests
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
PDF_STORAGE_PATH = "policy.pdf"
CHAIN_CACHE: Dict[str, RetrievalQA] = {}
CONCURRENCY_LIMIT = 3 # A safe number of concurrent requests to avoid rate limits

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
You are an AI expert specializing in insurance policy analysis. Your task is to provide a clear and accurate answer based *exclusively* on the provided text excerpts from a policy document.
Use the following context to answer the user's question. Synthesize the information from all provided chunks to form a complete answer.
If the information is present, provide a direct and factual answer.
If the context does not contain the answer, state: "The answer to this question could not be determined from the provided document excerpts."

Context:
{context}

Question:
{question}

Factual and Precise Answer:
"""

def create_and_cache_qa_chain(doc_url: str):
    """
    Creates a simple, robust QA chain and caches it.
    Uses a standard retriever with a larger context window.
    """
    if doc_url in CHAIN_CACHE:
        print(f"[INFO] Using cached QA chain for document: {doc_url}")
        return CHAIN_CACHE[doc_url]

    print(f"[INFO] New document. Creating simple & robust QA chain for: {doc_url}")
    
    response = requests.get(doc_url)
    response.raise_for_status()
    with open(PDF_STORAGE_PATH, 'wb') as f:
        f.write(response.content)
    
    loader = PyPDFLoader(PDF_STORAGE_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Use a basic retriever but fetch more documents (k=8) for better context.
    retriever = vectorstore.as_retriever(search_kwargs={'k': 8})

    # Critically, disable LangChain's automatic retries to prevent retry storms.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,
        max_retries=0 # Try only once.
    )
    
    prompt = PromptTemplate(template=PROMPT_V_EXPERT, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )
    
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] Simple & robust QA chain for {doc_url} created and cached.")
    
    if os.path.exists(PDF_STORAGE_PATH):
        os.remove(PDF_STORAGE_PATH)
        
    return qa_chain

async def process_question_with_semaphore(qa_chain, question, semaphore):
    async with semaphore:
        # A tiny delay helps smooth out the initial burst of requests
        await asyncio.sleep(0.2)
        return await qa_chain.ainvoke({"query": question})

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(req: HackRxRequest):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")
    
    try:
        qa_chain = create_and_cache_qa_chain(req.documents)
        
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = [process_question_with_semaphore(qa_chain, q, semaphore) for q in req.questions]
        
        results = await asyncio.gather(*tasks)
        answers = [res["result"] for res in results]
            
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running the definitive version for Level 2."}