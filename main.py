import os
import requests
import asyncio
import re # Import the regular expressions library
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
REQUEST_DELAY = 1.1 # The slow-but-reliable delay to avoid rate limits

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
1. Read all context excerpts carefully.
2. Form a comprehensive answer. If there are conflicting clauses (e.g., a general coverage and a specific exclusion), prioritize the most specific clause to make a final judgment.
3. Provide a direct, factual answer to the question.
4. If, and only if, the information to answer the question is not present in the context, state: "The answer to this question could not be determined from the provided document excerpts."

Context:
{context}

Question:
{question}

Definitive Answer:
"""

# CRITICAL FIX #1: The Text Pre-processing Function
def preprocess_text(documents):
    """
    Cleans the raw text extracted from the PDF to remove common artifacts.
    """
    cleaned_docs = []
    for doc in documents:
        # Get the raw text
        text = doc.page_content
        
        # Remove repeating headers (case-insensitive)
        text = re.sub(r'HDFC ERGO General Insurance Company Limited', '', text, flags=re.IGNORECASE)
        # Remove page numbers and footers like "1 | Page" or "Page 1 of 39"
        text = re.sub(r'\d+\s*\|\s*Page', '', text)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text)
        # Remove the UIN (policy number) that appears in footers
        text = re.sub(r'UIN: \w+', '', text)
        # Collapse multiple newlines into a single one to preserve paragraphs
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        # Update the document content
        doc.page_content = text
        cleaned_docs.append(doc)
    return cleaned_docs

def create_and_cache_qa_chain(doc_url: str):
    """
    Creates the definitive QA chain using pre-processing and ParentDocumentRetriever.
    """
    if doc_url in CHAIN_CACHE:
        print(f"[INFO] Using cached QA chain for document: {doc_url}")
        return CHAIN_CACHE[doc_url]

    print(f"[INFO] New document. Creating Phoenix QA chain for: {doc_url}")
    
    response = requests.get(doc_url)
    response.raise_for_status()
    with open(PDF_STORAGE_PATH, 'wb') as f:
        f.write(response.content)
    
    loader = PyPDFLoader(PDF_STORAGE_PATH)
    raw_docs = loader.load()

    # Apply the critical pre-processing step
    cleaned_docs = preprocess_text(raw_docs)

    # ACCURACY FIX: Use ParentDocumentRetriever on the CLEANED text
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    vectorstore = FAISS.from_documents(cleaned_docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter
    )
    retriever.add_documents(cleaned_docs)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = PromptTemplate(template=PROMPT_V_FINAL, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs
    )
    
    CHAIN_CACHE[doc_url] = qa_chain
    print(f"[INFO] Phoenix QA chain for {doc_url} created and cached.")
    
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
        
        # LATENCY FIX: Use the slow but 100% reliable sequential loop
        answers = []
        for question in req.questions:
            result = await qa_chain.ainvoke({"query": question})
            answers.append(result["result"])
            await asyncio.sleep(REQUEST_DELAY)
            
    except Exception as e:
        print(f"[CRITICAL ERROR] An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    return HackRxResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "API is running the definitive Phoenix version."}