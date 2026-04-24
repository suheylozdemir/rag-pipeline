from fastapi import FastAPI, UploadFile, File
from app.rag import create_rag_pipeline, ask_question
import tempfile
import os

app = FastAPI()

qa_chain = None  # pipeline starts empty

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global qa_chain
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create RAG pipeline from the file
    qa_chain = create_rag_pipeline(tmp_path)
    os.unlink(tmp_path)  # delete temp file
    
    return {"message": "Document uploaded successfully"}

@app.post("/ask")
async def ask(question: str):
    if qa_chain is None:
        return {"error": "No document uploaded yet"}
    
    answer = ask_question(qa_chain, question)
    return {"question": question, "answer": answer}