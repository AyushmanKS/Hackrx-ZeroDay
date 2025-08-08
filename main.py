import os
import requests
import asyncio # <-- Import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
HACKRX_TOKEN = "ed219fa46cac4026e20b9562bdfeecd13373a1e0a7529dee5b196421a3d97d4d"
PDF_STORAGE_PATH = "policy.pdf"

# --- API Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App Initialization ---
app = FastAPI()
auth_scheme = HTTPBearer()

# --- Security ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- Core Logic ---
def create_retrieval_qa_chain(pdf_path: str):
    # This part doesn't need to be async, it runs once per request.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(texts, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    # IMPORTANT: The chain now supports asynchronous calls with .ainvoke
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )
    return qa_chain

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(req: HackRxRequest, token: str = Depends(verify_token)):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")
    
    document_url = req.documents
    questions = req.questions
    
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        with open(PDF_STORAGE_PATH, 'wb') as f:
            f.write(response.content)

        qa_chain = create_retrieval_qa_chain(PDF_STORAGE_PATH)

        # --- ASYNCHRONOUS QUESTION PROCESSING ---
        # 1. Create a list of asynchronous tasks. .ainvoke is the async version of .invoke
        tasks = [qa_chain.ainvoke({"query": q}) for q in questions]
        
        # 2. Run all tasks concurrently and wait for them all to complete.
        results = await asyncio.gather(*tasks)
        
        # 3. Extract the answer text from each result.
        answers = [res["result"] for res in results]
        
    except Exception as e:
        # It's useful to print the error to the log for debugging
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(PDF_STORAGE_PATH):
            os.remove(PDF_STORAGE_PATH)

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /api/v1/hackrx/run"}