import os
import requests
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
    print("--- Inside create_retrieval_qa_chain ---")
    
    # 1. Load the document
    print("[DEBUG] Step 1: Loading PDF with PyPDFLoader...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print("[DEBUG] PDF loaded successfully.")

    # 2. Split the document into chunks
    print("[DEBUG] Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"[DEBUG] Document split into {len(texts)} chunks.")

    # 3. Create embeddings and vector store (FAISS)
    print("[DEBUG] Step 3: Creating embeddings and FAISS index... (This is the most memory-intensive step)")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(texts, embeddings)
    print("[DEBUG] FAISS index created successfully.")

    # 4. Create the Google Gemini LLM and the QA chain
    print("[DEBUG] Step 4: Creating the QA Chain with Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )
    print("[DEBUG] QA Chain created successfully.")
    print("--- Chain creation complete. ---")
    return qa_chain

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(req: HackRxRequest, token: str = Depends(verify_token)):
    print("\n--- Received new request ---")
    if not os.environ.get("GOOGLE_API_KEY"):
        print("[ERROR] GOOGLE_API_KEY not found!")
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")
    
    print("[DEBUG] GOOGLE_API_KEY found.")
    document_url = req.documents
    questions = req.questions
    
    try:
        print(f"[DEBUG] Downloading document from: {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()
        
        with open(PDF_STORAGE_PATH, 'wb') as f:
            f.write(response.content)
        print("[DEBUG] Document downloaded and saved.")

        qa_chain = create_retrieval_qa_chain(PDF_STORAGE_PATH)

        answers = []
        print(f"[DEBUG] Starting to process {len(questions)} questions...")
        for i, question in enumerate(questions):
            print(f"[DEBUG] Processing question {i+1}: {question}")
            result = qa_chain.invoke({"query": question})
            answers.append(result["result"])
            print(f"[DEBUG] Answer for question {i+1} generated.")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(PDF_STORAGE_PATH):
            os.remove(PDF_STORAGE_PATH)
        print("--- Request finished. ---")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /api/v1/hackrx/run"}
