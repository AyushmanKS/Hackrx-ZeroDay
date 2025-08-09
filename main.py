import os
import requests
import asyncio
from fastapi import FastAPI
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
HACKRX_TOKEN = "ed219fa46cac4026e20b9562bdfeecd13373a1e0a7529dee5b196421a3d97d4d"
PDF_STORAGE_PATH = "policy.pdf"
CHAIN_CACHE: Dict[str, RetrievalQA] = {}
REQUEST_DELAY = 1.1 

# --- API Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Core Logic ---

# Refined "Expert" Prompt
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
    Creates a high-accuracy QA chain using smaller chunks and MMR search.
    """
    if doc_url in CHAIN_CACHE:
        print(f"[INFO] Using cached QA chain for document: {doc_url}")
        return CHAIN_CACHE[doc_url]

    print(f"[INFO] New document. Creating optimized QA chain for: {doc_url}")
    
    response = requests.get(doc_url)
    response.raise_for_status()
    with open(PDF_STORAGE_PATH, 'wb') as f:
        f.write(response.content)
    
    loader = PyPDFLoader(PDF_STORAGE_PATH)
    documents = loader.load()

    # STRATEGY 1: Smaller, more granular chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"[INFO] Document split into {len(texts)} smaller chunks.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(texts, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # STRATEGY 2: Use Maximal Marginal Relevance (MMR) search
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 8, 'fetch_k': 20} # Fetch 20 docs, then use MMR to pick the best 8.
    )
    
    # STRATEGY 3: Use the refined expert prompt
    prompt = PromptTemplate(template=PROMPT_V_EXPERT, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False
    )
    
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] Optimized QA chain for {doc_url} created and cached.")
    
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
        
        answers = []
        for question in req.questions:
            result = await qa_chain.ainvoke({"query": question})
            answers.append(result["result"])
            await asyncio.sleep(REQUEST_DELAY)
            
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running with Level 2 accuracy enhancements."}