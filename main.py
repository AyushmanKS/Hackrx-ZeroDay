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
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

# --- Configuration ---
PDF_STORAGE_PATH = "policy.pdf"
CHAIN_CACHE: Dict[str, RetrievalQA] = {}
CONCURRENCY_LIMIT = 3 # A safe number of concurrent requests

# --- API Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Core Logic ---

PROMPT_V_FINAL = """
You are an AI Policy Adjudicator. Your task is to act as an expert and provide a definitive answer to a question about a policy document based *exclusively* on the provided context.
The context may contain multiple, sometimes conflicting, clauses. Your job is to synthesize them.
1.  Read all context excerpts carefully.
2.  Form a comprehensive answer. If there are conflicting clauses (e.g., a general coverage and a specific exclusion), prioritize the most specific clause to make a final judgment.
3.  Provide a direct, factual answer to the question.
4.  If, and only if, the information to answer the question is not present in the context, state: "The answer to this question could not be determined from the provided document excerpts."

Context:
{context}

Question:
{question}

Definitive Answer:
"""

def create_and_cache_qa_chain(doc_url: str):
    """
    Creates the definitive balanced QA chain and caches it.
    Uses Parent Document Retriever for rich context.
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

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    vectorstore = FAISS.from_documents(docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter
    )
    retriever.add_documents(docs)

    # THE CRITICAL BUG FIX: Explicitly disable LangChain's automatic retries.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,
        max_retries=0 # Try only once. Prevents the retry storm.
    )
    
    prompt = PromptTemplate(template=PROMPT_V_FINAL, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs
    )
    
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] Definitive QA chain for {doc_url} created and cached.")
    
    if os.path.exists(PDF_STORAGE_PATH):
        os.remove(PDF_STORAGE_PATH)
        
    return qa_chain

async def process_question_with_semaphore(qa_chain, question, semaphore):
    async with semaphore:
        # A small delay *before* the call helps prevent bursting the API limit
        await asyncio.sleep(0.5) 
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
        # Return a more informative error message to Postman
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running the final, stable version."}