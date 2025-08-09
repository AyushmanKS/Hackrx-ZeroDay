# Hackrx-ZeroDay: LLM-Powered Intelligent Query–Retrieval System

This project is a submission for the Hackrx hackathon. It is an API that uses a Large Language Model to answer natural language questions about a provided document.

---

### **Live Webhook URL**

The fully deployed and functional API endpoint is:
`https://hackrx-zeroday-production.up.railway.app/api/v1/hackrx/run`

---

### **Tech Stack**

*   **Backend Framework:** FastAPI
*   **LLM:** Google Gemini 1.5 Flash
*   **Vector Search:** FAISS (in-memory)
*   **Deployment Platform:** Railway

---

### **Design & Workflow**

The application follows a classic Retrieval-Augmented Generation (RAG) pattern:

1.  **API Endpoint:** A FastAPI server receives a POST request containing a document URL and a list of questions.
2.  **Document Ingestion:** The system downloads the PDF from the URL.
3.  **Chunking:** The document text is split into smaller, manageable chunks.
4.  **Embedding:** Each chunk is converted into a numerical vector using Google's embedding model. This vector represents the semantic meaning of the text.
5.  **Indexing:** The vectors are stored in an in-memory FAISS index for extremely fast semantic search.
6.  **Retrieval:** For each question, the system finds the most relevant text chunks from the document by searching the FAISS index.
7.  **Answer Generation:** The original question and the retrieved text chunks are passed to the Gemini LLM with a prompt instructing it to answer based *only* on the provided context.
8.  **Performance:** To handle the 10-question test case without triggering API rate limits or server timeouts, the system processes questions sequentially with a small delay between each call, ensuring both speed and reliability.
9.  **Response:** The final answers are collected and returned in the required JSON format.
