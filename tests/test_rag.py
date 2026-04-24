from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ask_without_upload():
    # Test asking without uploading a document first
    response = client.post("/ask?question=What is ASX?")
    assert response.status_code == 200
    assert response.json()["error"] == "No document uploaded yet"

@patch("app.rag.OpenAIEmbeddings")
@patch("app.rag.ChatOpenAI")
@patch("app.rag.Chroma")
def test_upload_and_ask(mock_chroma, mock_llm, mock_embeddings):
    # Mock the RAG pipeline components
    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = {"result": "ASX is the Australian Securities Exchange"}
    
    with patch("app.rag.RetrievalQA") as mock_retrieval:
        mock_retrieval.from_chain_type.return_value = mock_qa_chain
        
        # Upload document
        with open("app/sample.txt", "rb") as f:
            response = client.post("/upload", files={"file": f})
        assert response.status_code == 200
        
        # Ask question
        response = client.post("/ask?question=What is ASX?")
        assert response.status_code == 200
        assert "answer" in response.json()